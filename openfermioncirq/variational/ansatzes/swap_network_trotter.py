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

"""A variational ansatz based on a linear swap network Trotter step."""

from typing import Match, Optional, Sequence, Tuple, cast

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
    0    1         2         3
    │    │         │         │
    XXYY─XXYY^T0_1 XXYY──────XXYY^T2_3
    │    │         │         │
    YXXY─#2^W0_1   YXXY──────#2^W2_3
    │    │         │         │
    @────@^V0_1    @─────────@^V2_3
    │    │         │         │
    ×ᶠ───×ᶠ        ×ᶠ────────×ᶠ
    │    │         │         │
    │    XXYY──────XXYY^T0_3 │
    │    │         │         │
    │    YXXY──────#2^W0_3   │
    │    │         │         │
    │    @─────────@^V0_3    │
    │    │         │         │
    │    ×ᶠ────────×ᶠ        │
    │    │         │         │
    XXYY─XXYY^T1_3 XXYY──────XXYY^T0_2
    │    │         │         │
    YXXY─#2^W1_3   YXXY──────#2^W0_2
    │    │         │         │
    @────@^V1_3    @─────────@^V0_2
    │    │         │         │
    ×ᶠ───×ᶠ        ×ᶠ────────×ᶠ
    │    │         │         │
    Z^U3 XXYY──────XXYY^T1_2 Z^U0
    │    │         │         │
    │    YXXY──────#2^W1_2   │
    │    │         │         │
    │    @─────────@^V1_2    │
    │    │         │         │
    │    ×ᶠ────────×ᶠ        │
    │    │         │         │
    │    Z^U2      Z^U1      │
    │    │         │         │
    │    @─────────@^V1_2    │
    │    │         │         │
    │    #2────────YXXY^W1_2 │
    │    │         │         │
    │    XXYY──────XXYY^T1_2 │
    │    │         │         │
    │    ×ᶠ────────×ᶠ        │
    │    │         │         │
    @────@^V1_3    @─────────@^V0_2
    │    │         │         │
    #2───YXXY^W1_3 #2────────YXXY^W0_2
    │    │         │         │
    XXYY─XXYY^T1_3 XXYY──────XXYY^T0_2
    │    │         │         │
    ×ᶠ───×ᶠ        ×ᶠ────────×ᶠ
    │    │         │         │
    │    @─────────@^V0_3    │
    │    │         │         │
    │    #2────────YXXY^W0_3 │
    │    │         │         │
    │    XXYY──────XXYY^T0_3 │
    │    │         │         │
    │    ×ᶠ────────×ᶠ        │
    │    │         │         │
    @────@^V0_1    @─────────@^V2_3
    │    │         │         │
    #2───YXXY^W0_1 #2────────YXXY^W2_3
    │    │         │         │
    XXYY─XXYY^T0_1 XXYY──────XXYY^T2_3
    │    │         │         │
    ×ᶠ───×ᶠ        ×ᶠ────────×ᶠ
    │    │         │         │
    ```

    The Hamiltonian associated with the ansatz determines which XXYY, YXXY, CZ,
    and Z gates are included. This basic template can be repeated, with each
    iteration introducing a new set of parameters.

    The default initial parameters of the ansatz are chosen
    so that the ansatz circuit consists of a sequence of second-order
    Trotter steps approximating the dynamics of the time-dependent
    Hamiltonian H(t) = T + (t/A)V, where T is the one-body term and V is
    the two-body term of the Hamiltonian used to generate the
    ansatz circuit, and t ranges from 0 to A and A is an adjustable value
    that defaults to the sum of the absolute values of the coefficients of
    the Jordan-Wigner transformed two-body operator V.
    The number of Trotter steps is equal to the number
    of iterations in the ansatz. This choice is motivated by the idea of
    state preparation via adiabatic evolution.
    The dynamics of H(t) are approximated as follows. First, the total
    evolution time of A is split into segments of length A / r, where r
    is the number of Trotter steps. Then, each Trotter step simulates H(t)
    for a time length of A / r, where t is the midpoint of the
    corresponding time segment. As an example, suppose A is 100 and the
    ansatz has two iterations. Then the approximation is achieved with two
    Trotter steps. The first Trotter step simulates H(25) for a time length
    of 50, and the second Trotter step simulates H(75) for a time length of 50.
    """

    def __init__(self,
                 hamiltonian: openfermion.DiagonalCoulombHamiltonian,
                 iterations: int=1,
                 include_all_xxyy: bool=False,
                 include_all_yxxy: bool=False,
                 include_all_cz: bool=False,
                 include_all_z: bool=False,
                 adiabatic_evolution_time: Optional[float]=None,
                 qubits: Optional[Sequence[cirq.QubitId]]=None
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
            adiabatic_evolution_time: The time scale for Hamiltonian evolution
                used to determine the default initial parameters of the ansatz.
                This is the value A from the docstring of this class.
                If not specified, defaults to the sum of the absolute values
                of the entries of the two-body tensor of the Hamiltonian.
            qubits: Qubits to be used by the ansatz circuit. If not specified,
                then qubits will automatically be generated by the
                `_generate_qubits` method.
        """
        self.hamiltonian = hamiltonian
        self.iterations = iterations
        self.include_all_xxyy = include_all_xxyy
        self.include_all_yxxy = include_all_yxxy
        self.include_all_cz = include_all_cz
        self.include_all_z = include_all_z

        if adiabatic_evolution_time is None:
            adiabatic_evolution_time = (
                    numpy.sum(numpy.abs(hamiltonian.two_body)))
        self.adiabatic_evolution_time = cast(float, adiabatic_evolution_time)

        super().__init__(qubits)

    def param_names(self) -> Sequence[str]:
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

    def param_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """Bounds on the parameters."""
        bounds = []
        for param_name in self.param_names():
            if param_name.startswith('U') or param_name.startswith('V'):
                bounds.append((-1.0, 1.0))
            elif param_name.startswith('T') or param_name.startswith('W'):
                bounds.append((-2.0, 2.0))
        return bounds

    def _generate_qubits(self) -> Sequence[cirq.QubitId]:
        return cirq.LineQubit.range(openfermion.count_qubits(self.hamiltonian))

    def operations(self, qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit."""
        # TODO implement asymmetric ansatzes?

        for i in range(self.iterations):

            suffix = '-{}'.format(i) if self.iterations > 1 else ''

            # Apply one- and two-body interactions with a swap network that
            # reverses the order of the modes
            def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
                if 'T{}_{}'.format(p, q) + suffix in self.params:
                    yield XXYYGate(half_turns=self.params[
                              'T{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'W{}_{}'.format(p, q) + suffix in self.params:
                    yield YXXYGate(half_turns=self.params[
                              'W{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'V{}_{}'.format(p, q) + suffix in self.params:
                    yield cirq.Rot11Gate(half_turns=self.params[
                              'V{}_{}'.format(p, q) + suffix]).on(a, b)
            yield swap_network(
                    qubits, one_and_two_body_interaction, fermionic=True)
            qubits = qubits[::-1]

            # Apply one-body potential
            yield (cirq.RotZGate(half_turns=
                       self.params['U{}'.format(p) + suffix]).on(qubits[p])
                   for p in range(len(qubits))
                   if 'U{}'.format(p) + suffix in self.params)

            # Apply one- and two-body interactions again. This time, reorder
            # them so that the entire iteration is symmetric
            def one_and_two_body_interaction_reversed_order(p, q, a, b
                    ) -> cirq.OP_TREE:
                if 'V{}_{}'.format(p, q) + suffix in self.params:
                    yield cirq.Rot11Gate(half_turns=self.params[
                              'V{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'W{}_{}'.format(p, q) + suffix in self.params:
                    yield YXXYGate(half_turns=self.params[
                              'W{}_{}'.format(p, q) + suffix]).on(a, b)
                if 'T{}_{}'.format(p, q) + suffix in self.params:
                    yield XXYYGate(half_turns=self.params[
                              'T{}_{}'.format(p, q) + suffix]).on(a, b)
            yield swap_network(
                    qubits, one_and_two_body_interaction_reversed_order,
                    fermionic=True, offset=True)
            qubits = qubits[::-1]

    def default_initial_params(self) -> numpy.ndarray:
        """Approximate evolution by H(t) = T + (t/A)V.

        Sets the parameters so that the ansatz circuit consists of a sequence
        of second-order Trotter steps approximating the dynamics of the
        time-dependent Hamiltonian H(t) = T + (t/A)V, where T is the one-body
        term and V is the two-body term of the Hamiltonian used to generate the
        ansatz circuit, and t ranges from 0 to A, where A is equal to
        `self.adibatic_evolution_time`. The number of Trotter steps
        is equal to the number of iterations in the ansatz. This choice is
        motivated by the idea of state preparation via adiabatic evolution.

        The dynamics of H(t) are approximated as follows. First, the total
        evolution time of A is split into segments of length A / r, where r
        is the number of Trotter steps. Then, each Trotter step simulates H(t)
        for a time length of A / r, where t is the midpoint of the
        corresponding time segment. As an example, suppose A is 100 and the
        ansatz has two iterations. Then the approximation is achieved with two
        Trotter steps. The first Trotter step simulates H(25) for a time length
        of 50, and the second Trotter step simulates H(75) for a time length
        of 50.
        """

        total_time = self.adiabatic_evolution_time
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
                    -hamiltonian.one_body[p, p].real * step_time / numpy.pi, 2))
            else:
                letter, p, q, i = cast(
                        Match, TWV_pattern.match(param_name)).groups()
                p, q, i = int(p), int(q), int(i) if i else 0
                # Use the midpoint of the time segment
                interpolation_progress = 0.5 * (2 * i + 1) / self.iterations
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
