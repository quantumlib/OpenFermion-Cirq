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

"""A variational ansatz based on a split-operator Trotter step."""

from typing import Iterable, Optional, Sequence, Tuple, cast

import itertools

import numpy

import cirq
import openfermion

from openfermioncirq import bogoliubov_transform, swap_network
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)


class SplitOperatorTrotterAnsatz(VariationalAnsatz):
    """An ansatz based on a split-operator Trotter step.

    This ansatz uses as a template the form of a second-order Trotter step
    based on the split-operator simulation method described in arXiv:1706.00023.
    The ansatz circuit and default initial parameters are determined by an
    instance of the DiagonalCoulombHamiltonian class.

    Example: The ansatz for a spinless jellium Hamiltonian on a 2x2 grid with
    one iteration has the circuit::

        0       1           2           3
        │       │           │           │
        │       │           YXXY────────#2^0.5
        │       │           │           │
        │       YXXY────────#2^0.608    │
        │       │           │           │
        │       │           YXXY────────#2^-0.333
        │       │           │           │
        YXXY────#2^0.667    │           │
        │       │           │           │
        Z       YXXY────────#2          │
        │       │           │           │
        Z       Z           YXXY────────#2^-0.392
        │       │           │           │
        │       Z^U_1_0     │           Z
        │       │           │           │
        │       Z           Z^U_2_0     Z^U_3_0
        │       │           │           │
        │       │           │           Z
        │       │           │           │
        │       │           YXXY────────#2^0.392
        │       │           │           │
        │       YXXY────────#2^-1       │
        │       │           │           │
        YXXY────#2^-0.667   │           │
        │       │           │           │
        │       │           YXXY────────#2^0.333
        │       │           │           │
        │       YXXY────────#2^-0.608   │
        │       │           │           │
        @───────@^V_0_1_0   │           │
        │       │           │           │
        ×───────×           YXXY────────#2^-0.5
        │       │           │           │
        │       │           @───────────@^V_2_3_0
        │       │           │           │
        │       │           ×───────────×
        │       │           │           │
        │       @───────────@^V_0_3_0   │
        │       │           │           │
        │       ×───────────×           │
        │       │           │           │
        @───────@^V_1_3_0   @───────────@^V_0_2_0
        │       │           │           │
        ×───────×           ×───────────×
        │       │           │           │
        │       @───────────@^V_1_2_0   │
        │       │           │           │
        │       ×───────────×           │
        │       │           │           │
        #2──────YXXY^0.5    │           │
        │       │           │           │
        │       #2──────────YXXY^0.608  │
        │       │           │           │
        #2──────YXXY^-0.333 │           │
        │       │           │           │
        │       │           #2──────────YXXY^0.667
        │       │           │           │
        │       #2──────────YXXY        Z
        │       │           │           │
        #2──────YXXY^-0.392 Z           Z
        │       │           │           │
        Z       │           Z^U_1_0     │
        │       │           │           │
        Z^U_3_0 Z^U_2_0     Z           │
        │       │           │           │
        Z       │           │           │
        │       │           │           │
        #2──────YXXY^0.392  │           │
        │       │           │           │
        │       #2──────────YXXY^-1     │
        │       │           │           │
        │       │           #2──────────YXXY^-0.667
        │       │           │           │
        #2──────YXXY^0.333  │           │
        │       │           │           │
        │       #2──────────YXXY^-0.608 │
        │       │           │           │
        #2──────YXXY^-0.5   │           │
        │       │           │           │

    This basic template can be repeated, with each iteration introducing a
    new set of parameters.

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
        self.include_all_cz = include_all_cz
        self.include_all_z = include_all_z

        if adiabatic_evolution_time is None:
            adiabatic_evolution_time = (
                    numpy.sum(numpy.abs(hamiltonian.two_body)))
        self.adiabatic_evolution_time = cast(float, adiabatic_evolution_time)

        quad_ham = openfermion.QuadraticHamiltonian(hamiltonian.one_body)
        # Get the basis change matrix that diagonalizes the one-body term
        # and associated orbital energies
        self.orbital_energies, self.basis_change_matrix, _ = (
                quad_ham.diagonalizing_bogoliubov_transform()
        )

        super().__init__(qubits)

    def params(self) -> Iterable[cirq.Symbol]:
        """The names of the parameters of the ansatz."""
        for i in range(self.iterations):
            for p in range(len(self.qubits)):
                if (self.include_all_z or not
                        numpy.isclose(self.orbital_energies[p], 0)):
                    yield LetterWithSubscripts('U', p, i)
            for p, q in itertools.combinations(range(len(self.qubits)), 2):
                if (self.include_all_cz or not
                        numpy.isclose(self.hamiltonian.two_body[p, q], 0)):
                    yield LetterWithSubscripts('V', p, q, i)

    def param_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """Bounds on the parameters."""
        return [(-1.0, 1.0)] * len(list(self.params()))

    def _generate_qubits(self) -> Sequence[cirq.QubitId]:
        """Produce qubits that can be used by the ansatz circuit."""
        return cirq.LineQubit.range(openfermion.count_qubits(self.hamiltonian))

    def operations(self, qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit."""
        # TODO implement asymmetric ansatz

        param_set = set(self.params())

        # Change to the basis in which the one-body term is diagonal
        yield cirq.inverse(
                bogoliubov_transform(qubits, self.basis_change_matrix))

        for i in range(self.iterations):

            # Simulate one-body terms
            for p in range(len(qubits)):
                u_symbol = LetterWithSubscripts('U', p, i)
                if u_symbol in param_set:
                    yield cirq.RotZGate(half_turns=u_symbol).on(qubits[p])

            # Rotate to the computational basis
            yield bogoliubov_transform(qubits, self.basis_change_matrix)

            # Simulate the two-body terms
            def two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
                v_symbol = LetterWithSubscripts('V', p, q, i)
                if v_symbol in param_set:
                    yield cirq.Rot11Gate(half_turns=v_symbol).on(a, b)
            yield swap_network(qubits, two_body_interaction)
            qubits = qubits[::-1]

            # Rotate back to the basis in which the one-body term is diagonal
            yield cirq.inverse(
                    bogoliubov_transform(qubits, self.basis_change_matrix))

            # Simulate one-body terms again
            for p in range(len(qubits)):
                u_symbol = LetterWithSubscripts('U', p, i)
                if u_symbol in param_set:
                    yield cirq.RotZGate(half_turns=u_symbol).on(qubits[p])

        # Rotate to the computational basis
        yield bogoliubov_transform(qubits, self.basis_change_matrix)

    def qubit_permutation(self, qubits: Sequence[cirq.QubitId]
                          ) -> Sequence[cirq.QubitId]:
        """The qubit permutation induced by the ansatz circuit."""
        # Every iteration reverses the qubit ordering due to the use of a
        # swap network
        if self.iterations & 1:
            return qubits[::-1]
        else:
            return qubits

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

        params = []
        for param in self.params():
            if param.letter == 'U':
                p, i = param.subscripts
                params.append(_canonicalize_exponent(
                    -0.5 * self.orbital_energies[p] * step_time / numpy.pi, 2))
            else:
                p, q, i = param.subscripts
                # Use the midpoint of the time segment
                interpolation_progress = 0.5 * (2 * i + 1) / self.iterations
                params.append(_canonicalize_exponent(
                    -2 * hamiltonian.two_body[p, q] * interpolation_progress *
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
