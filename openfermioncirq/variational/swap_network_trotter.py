#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import List, Match, Tuple, cast

import itertools
import re

import numpy

import cirq
import openfermion

from openfermioncirq import XXYYGate, YXXYGate, swap_network
from openfermioncirq.variational.ansatz import VariationalAnsatz


class SwapNetworkTrotterAnsatz(VariationalAnsatz):
    """An ansatz based on the fermionic swap network.

    This ansatz uses as a template the form of a second-order Trotter step
    based on the "fermionic swap network" described in arXiv:1711.04789.
    The ansatz circuit and default initial parameters are determined by an
    instance of the DiagonalCoulombHamiltonian class.

    Example: The ansatz on 4 qubits with one iteration and all gates included
    has the circuit

    ```
    0         1         2         3
    │         │         │         │
    XXYY^T0_1─XXYY      XXYY^T2_3─XXYY
    │         │         │         │
    YXXY^W0_1─#2        YXXY^W2_3─#2
    │         │         │         │
    @^V0_1────Z         @^V2_3────Z
    │         │         │         │
    ×ᶠ────────×ᶠ        ×ᶠ────────×ᶠ
    │         │         │         │
    │         XXYY^T0_3─XXYY      │
    │         │         │         │
    │         YXXY^W0_3─#2        │
    │         │         │         │
    │         @^V0_3────Z         │
    │         │         │         │
    │         ×ᶠ────────×ᶠ        │
    │         │         │         │
    XXYY^T1_3─XXYY      XXYY^T0_2─XXYY
    │         │         │         │
    YXXY^W1_3─#2        YXXY^W0_2─#2
    │         │         │         │
    @^V1_3────Z         @^V0_2────Z
    │         │         │         │
    ×ᶠ────────×ᶠ        ×ᶠ────────×ᶠ
    │         │         │         │
    Z^U3      XXYY^T1_2─XXYY      Z^U0
    │         │         │         │
    │         YXXY^W1_2─#2        │
    │         │         │         │
    │         @^V1_2────Z         │
    │         │         │         │
    │         ×ᶠ────────×ᶠ        │
    │         │         │         │
    │         Z^U2      Z^U1      │
    │         │         │         │
    │         XXYY^T1_2─XXYY      │
    │         │         │         │
    │         #2^W1_2───YXXY      │
    │         │         │         │
    │         Z^V1_2────@         │
    │         │         │         │
    │         ×ᶠ────────×ᶠ        │
    │         │         │         │
    XXYY^T1_3─XXYY      XXYY^T0_2─XXYY
    │         │         │         │
    #2^W1_3───YXXY      #2^W0_2───YXXY
    │         │         │         │
    Z^V1_3────@         Z^V0_2────@
    │         │         │         │
    ×ᶠ────────×ᶠ        ×ᶠ────────×ᶠ
    │         │         │         │
    │         XXYY^T0_3─XXYY      │
    │         │         │         │
    │         #2^W0_3───YXXY      │
    │         │         │         │
    │         Z^V0_3────@         │
    │         │         │         │
    │         ×ᶠ────────×ᶠ        │
    │         │         │         │
    XXYY^T0_1─XXYY      XXYY^T2_3─XXYY
    │         │         │         │
    #2^W0_1───YXXY      #2^W2_3───YXXY
    │         │         │         │
    Z^V0_1────@         Z^V2_3────@
    │         │         │         │
    ×ᶠ────────×ᶠ        ×ᶠ────────×ᶠ
    │         │         │         │
    ```

    The Hamiltonian associated with the ansatz determines which XXYY, YXXY, CZ,
    and Z gates are included. This basic template can be repeated, with each
    iteration coming with a new set of parameters.
    """

    def __init__(self,
                 hamiltonian: openfermion.DiagonalCoulombHamiltonian,
                 iterations: int=1,
                 include_all_xxyy: bool=False,
                 include_all_yxxy: bool=False,
                 include_all_cz: bool=False,
                 include_all_z: bool=False
                 ) -> None:
        """
        Args:
            hamiltonian: The Hamiltonian used to generate the ansatz
                circuit and default initial parameters.
            iterations: The number of iterations of the basic template to
                include in the circuit. The number of parameters grows linearly
                with this value.
            include_all_xxyy: Whether to include all possible XXYY-type
                parameterized gates in the ansatz (irrespective of the ansatz
                Hamiltonian)
            include_all_yxxy: Whether to include all possible YXXY-type
                parameterized gates in the ansatz (irrespective of the ansatz
                Hamiltonian)
            include_all_cz: Whether to include all possible CZ-type
                parameterized gates in the ansatz (irrespective of the ansatz
                Hamiltonian)
            include_all_z: Whether to include all possible Z-type
                parameterized gates in the ansatz (irrespective of the ansatz
                Hamiltonian)
        """
        self.hamiltonian = hamiltonian
        self.iterations = iterations
        self.include_all_xxyy = include_all_xxyy
        self.include_all_yxxy = include_all_yxxy
        self.include_all_cz = include_all_cz
        self.include_all_z = include_all_z
        self.qubits = cirq.LineQubit.range(
                openfermion.count_qubits(hamiltonian))
        super().__init__()

    def param_names(self) -> List[str]:
        """The names of the parameters of the ansatz."""
        names = []
        for i in range(self.iterations):
            suffix = '-{}'.format(i) if self.iterations > 1 else ''
            for p in range(len(self.qubits)):
                if (self.include_all_z or not
                        numpy.isclose(self.hamiltonian.one_body[p, p], 0)):
                    names.append('U{}'.format(p) + suffix)
            for p, q in itertools.combinations(range(len(self.qubits)), 2):
                if (self.include_all_xxyy or not
                        numpy.isclose(self.hamiltonian.one_body[p, q].real, 0)):
                    names.append('T{}_{}'.format(p, q) + suffix)
                if (self.include_all_yxxy or not
                        numpy.isclose(self.hamiltonian.one_body[p, q].imag, 0)):
                    names.append('W{}_{}'.format(p, q) + suffix)
                if (self.include_all_cz or not
                        numpy.isclose(self.hamiltonian.two_body[p, q], 0)):
                    names.append('V{}_{}'.format(p, q) + suffix)
        return names

    def param_bounds(self) -> List[Tuple[float, float]]:
        """Bounds on the parameters."""
        bounds = []
        for param_name in self.param_names():
            if param_name.startswith('U') or param_name.startswith('V'):
                bounds.append((-1.0, 1.0))
            elif param_name.startswith('T') or param_name.startswith('W'):
                bounds.append((-2.0, 2.0))
        return bounds

    def generate_circuit(self) -> cirq.Circuit:
        """Produce the ansatz circuit."""
        # TODO implement asymmetric ansatzes?

        qubits = self.qubits
        circuit_ = cirq.Circuit()

        for i in range(self.iterations):

            suffix = '-{}'.format(i) if self.iterations > 1 else ''

            def one_and_two_body_interaction(p, q, a, b):
                # Apply one- and two-body interactions to modes p and q
                # represented by qubits a and b
                if 'T{}_{}'.format(p, q) + suffix in self.params:
                    yield XXYYGate(quarter_turns=self.params[
                              'T{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'W{}_{}'.format(p, q) + suffix in self.params:
                    yield YXXYGate(quarter_turns=self.params[
                              'W{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'V{}_{}'.format(p, q) + suffix in self.params:
                    yield cirq.Rot11Gate(half_turns=self.params[
                              'V{}_{}'.format(p, q) + suffix]).on(a, b)

            # Apply one- and two-body interactions with a swap network that
            # reverses the order of the modes
            circuit_.append(
                    swap_network(
                        qubits, one_and_two_body_interaction, fermionic=True),
                    strategy=cirq.InsertStrategy.EARLIEST)
            qubits = qubits[::-1]

            # Apply one-body potential
            circuit_.append(
                    (cirq.RotZGate(half_turns=self.params[
                        'U{}'.format(p) + suffix]).on(qubits[p])
                     for p in range(len(qubits))
                     if 'U{}'.format(p) + suffix in self.params),
                    strategy=cirq.InsertStrategy.EARLIEST)

            # Apply the same one- and two-body interactions again
            circuit_.append(
                    swap_network(
                        qubits, one_and_two_body_interaction, fermionic=True,
                        offset=True),
                    strategy=cirq.InsertStrategy.EARLIEST)
            qubits = qubits[::-1]

        return circuit_

    def default_initial_params(self) -> numpy.ndarray:
        """Approximate adiabatic evolution by H(t) = T + (t/100)V.

        Sets the parameters so that the ansatz circuit consists of a sequence
        of second-order Trotter steps approximating the dynamics of the
        time-dependent Hamiltonian H(t) = T + (t/100)V, where T is the one-body
        term and V is the two-body term of the Hamiltonian used to generate the
        ansatz circuit, and t ranges from 0 to 100. The number of Trotter steps
        is equal to the number of iterations in the ansatz.
        """
        # TODO use midpoint for exponential

        total_time = 100
        step_time = total_time / self.iterations
        hamiltonian = self.hamiltonian

        U_pattern = re.compile('U([0-9]*)-?([0-9]*)?')
        TWV_pattern = re.compile('(T|W|V)([0-9]*)_([0-9]*)-?([0-9]*)?')

        params = []
        for param_name in self.param_names():
            if param_name.startswith('U'):
                p, i = cast(Match, U_pattern.match(param_name)).groups()
                p, i = int(p), int(i) if i else 0
                params.append(_canonicalize_exponent(
                    numpy.real_if_close(hamiltonian.one_body[p, p]) *
                    step_time /numpy.pi, 2))
            else:
                letter, p, q, i = cast(
                        Match, TWV_pattern.match(param_name)).groups()
                p, q, i = int(p), int(q), int(i) if i else 0
                interpolation_progress = (i + 1) / self.iterations
                if letter == 'T':
                    params.append(_canonicalize_exponent(
                        hamiltonian.one_body[p, q].real *
                        step_time / numpy.pi, 4))
                elif letter == 'W':
                    params.append(_canonicalize_exponent(
                        hamiltonian.one_body[p, q].imag *
                        step_time / numpy.pi, 4))
                elif letter == 'V':
                    params.append(_canonicalize_exponent(
                        -hamiltonian.two_body[p, q] * interpolation_progress *
                        step_time / numpy.pi, 2))

        return numpy.array(params)


def _canonicalize_exponent(exponent: float, period: int) -> float:
    # Shift into [-p/2, +p/2).
    exponent += period / 2
    exponent %= period
    exponent -= period / 2
    # Prefer (-p/2, +p/2] over [-p/2, +p/2).
    if exponent <= -period / 2:
        exponent += period  # coverage: ignore
    return exponent
