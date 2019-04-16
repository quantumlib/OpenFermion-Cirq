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

"""A variational ansatz based on a low rank Trotter step."""

from typing import Iterable, Optional, Sequence, TYPE_CHECKING, Tuple, cast

import itertools

import numpy
import sympy

import cirq
import openfermion

from openfermioncirq import bogoliubov_transform, swap_network
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import (
        LetterWithSubscripts)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


class LowRankTrotterAnsatz(VariationalAnsatz):
    """An ansatz based on a low rank Trotter step.

    This ansatz uses as a template the form of a second-order Trotter step
    based on the low rank simulation method described in arXiv:1808.02625.
    The ansatz circuit and default initial parameters are determined by an
    instance of the InteractionOperator class.

    Example: The ansatz for an LiH molecule in a minimal basis with one
    iteration and final rank set to 1 has the circuit::

        0         1           2           3
        │         │           │           │
        Z         Z           Z           Z
        │         │           │           │
        │         YXXY────────#2^-1       │
        │         │           │           │
        YXXY──────#2^0.0813   │           │
        │         │           │           │
        Z^U_0_0   │           YXXY────────#2^-0.0813
        │         │           │           │
        │         YXXY────────#2^-1       │
        │         │           │           │
        │         Z^U_1_0     │           Z^U_3_0
        │         │           │           │
        │         │           Z^U_2_0     │
        │         │           │           │
        │         YXXY────────#2^-1       │
        │         │           │           │
        YXXY──────#2^-0.051   │           │
        │         │           │           │
        │         │           YXXY────────#2^0.051
        │         │           │           │
        │         YXXY────────#2^-1       │
        │         │           │           │
        @─────────@^V_0_1_0_0 │           │
        │         │           │           │
        ×─────────×           @───────────@^V_2_3_0_0
        │         │           │           │
        │         │           ×───────────×
        │         │           │           │
        │         @───────────@^V_0_3_0_0 │
        │         │           │           │
        │         ×───────────×           │
        │         │           │           │
        @─────────@^V_1_3_0_0 @───────────@^V_0_2_0_0
        │         │           │           │
        ×─────────×           ×───────────×
        │         │           │           │
        Z^U_3_0_0 @───────────@^V_1_2_0_0 Z^U_0_0_0
        │         │           │           │
        Z         ×───────────×           Z
        │         │           │           │
        │         Z^U_2_0_0   Z^U_1_0_0   │
        │         │           │           │
        │         #2──────────YXXY^-1     │
        │         │           │           │
        │         │           #2──────────YXXY^0.132
        │         │           │           │
        #2────────YXXY^-0.132 │           │
        │         │           │           │
        │         #2──────────YXXY^-1     │
        │         │           │           │
    """

    def __init__(self,
                 hamiltonian: openfermion.InteractionOperator,
                 iterations: int=1,
                 final_rank: Optional[int]=None,
                 include_all_cz: bool=False,
                 include_all_z: bool=False,
                 adiabatic_evolution_time: Optional[float]=None,
                 spin_basis: bool=True,
                 qubits: Optional[Sequence[cirq.Qid]]=None
                 ) -> None:
        """
        Args:
            hamiltonian: The Hamiltonian used to generate the ansatz
                circuit and default initial parameters.
            iterations: The number of iterations of the basic template to
                include in the circuit. The number of parameters grows linearly
                with this value.
            final_rank: The rank at which to truncate the decomposition.
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
            spin_basis: Whether the Hamiltonian is given in the spin orbital
                (rather than spatial orbital) basis.
            qubits: Qubits to be used by the ansatz circuit. If not specified,
                then qubits will automatically be generated by the
                `_generate_qubits` method.
        """
        self.hamiltonian = hamiltonian
        self.iterations = iterations
        self.final_rank = final_rank
        self.include_all_cz = include_all_cz
        self.include_all_z = include_all_z

        if adiabatic_evolution_time is None:
            adiabatic_evolution_time = (
                    numpy.sum(numpy.abs(hamiltonian.two_body_tensor)))
        self.adiabatic_evolution_time = cast(float, adiabatic_evolution_time)

        # Perform the low rank decomposition of two-body operator.
        self.eigenvalues, one_body_squares, self.one_body_correction, _ = (
            openfermion.low_rank_two_body_decomposition(
                hamiltonian.two_body_tensor,
                final_rank=self.final_rank,
                spin_basis=spin_basis))

        # Get scaled density-density terms and basis transformation matrices.
        self.scaled_density_density_matrices = []  # type: List[numpy.ndarray]
        self.basis_change_matrices = []            # type: List[numpy.ndarray]
        for j in range(len(self.eigenvalues)):
            density_density_matrix, basis_change_matrix = (
                openfermion.prepare_one_body_squared_evolution(
                    one_body_squares[j]))
            self.scaled_density_density_matrices.append(
                    numpy.real(self.eigenvalues[j] * density_density_matrix))
            self.basis_change_matrices.append(basis_change_matrix)

        # Get transformation matrix and orbital energies for one-body terms
        one_body_coefficients = (
                hamiltonian.one_body_tensor + self.one_body_correction)
        quad_ham = openfermion.QuadraticHamiltonian(one_body_coefficients)
        self.one_body_energies, self.one_body_basis_change_matrix, _ = (
                quad_ham.diagonalizing_bogoliubov_transform()
        )

        super().__init__(qubits)

    def params(self) -> Iterable[sympy.Symbol]:
        """The parameters of the ansatz."""

        for i in range(self.iterations):

            for p in range(len(self.qubits)):
                # One-body energies
                if (self.include_all_z or not numpy.isclose(
                        self.one_body_energies[p], 0)):
                    yield LetterWithSubscripts('U', p, i)
                # Diagonal two-body coefficients for each singular vector
                for j in range(len(self.eigenvalues)):
                    two_body_coefficients = (
                            self.scaled_density_density_matrices[j])
                    if (self.include_all_z or not numpy.isclose(
                            two_body_coefficients[p, p], 0)):
                        yield LetterWithSubscripts('U', p, j, i)

            for p, q in itertools.combinations(range(len(self.qubits)), 2):
                # Off-diagonal two-body coefficients for each singular vector
                for j in range(len(self.eigenvalues)):
                    two_body_coefficients = (
                            self.scaled_density_density_matrices[j])
                    if (self.include_all_cz or not numpy.isclose(
                            two_body_coefficients[p, q], 0)):
                        yield LetterWithSubscripts('V', p, q, j, i)

    def param_bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """Bounds on the parameters."""
        return [(-1.0, 1.0)] * len(list(self.params()))

    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        """Produce qubits that can be used by the ansatz circuit."""
        return cirq.LineQubit.range(openfermion.count_qubits(self.hamiltonian))

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Produce the operations of the ansatz circuit."""

        n_qubits = len(qubits)
        param_set = set(self.params())

        for i in range(self.iterations):

            # Change to the basis in which the one-body term is diagonal
            yield bogoliubov_transform(
                    qubits, self.one_body_basis_change_matrix.T.conj())

            # Simulate the one-body terms.
            for p in range(n_qubits):
                u_symbol = LetterWithSubscripts('U', p, i)
                if u_symbol in param_set:
                    yield cirq.ZPowGate(exponent=u_symbol).on(qubits[p])

            # Simulate each singular vector of the two-body terms.
            prior_basis_matrix = self.one_body_basis_change_matrix

            for j in range(len(self.eigenvalues)):

                # Get the basis change matrix.
                basis_change_matrix = self.basis_change_matrices[j]

                # Merge previous basis change matrix with the inverse of the
                # current one
                merged_basis_change_matrix = numpy.dot(
                        prior_basis_matrix,
                        basis_change_matrix.T.conj())
                yield bogoliubov_transform(qubits, merged_basis_change_matrix)

                # Simulate the off-diagonal two-body terms.
                def two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
                    v_symbol = LetterWithSubscripts('V', p, q, j, i)
                    if v_symbol in param_set:
                        yield cirq.CZPowGate(exponent=v_symbol).on(a, b)
                yield swap_network(qubits, two_body_interaction)
                qubits = qubits[::-1]

                # Simulate the diagonal two-body terms.
                for p in range(n_qubits):
                    u_symbol = LetterWithSubscripts('U', p, j, i)
                    if u_symbol in param_set:
                        yield cirq.ZPowGate(exponent=u_symbol).on(qubits[p])

                # Update prior basis change matrix.
                prior_basis_matrix = basis_change_matrix

            # Undo final basis transformation.
            yield bogoliubov_transform(qubits, prior_basis_matrix)

    def qubit_permutation(self, qubits: Sequence[cirq.Qid]
                          ) -> Sequence[cirq.Qid]:
        """The qubit permutation induced by the ansatz circuit."""
        # An odd number of swap networks reverses the qubit ordering
        if self.iterations & 1 and len(self.eigenvalues) & 1:
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

        params = []

        for param in self.params():

            i = param.subscripts[-1]
            # Use the midpoint of the time segment
            interpolation_progress = 0.5 * (2 * i + 1) / self.iterations

            # One-body term
            if param.letter == 'U' and len(param.subscripts) == 2:
                p, _ = param.subscripts

                one_body_coefficients = (
                        self.hamiltonian.one_body_tensor
                        + interpolation_progress * self.one_body_correction)
                quad_ham = openfermion.QuadraticHamiltonian(
                        one_body_coefficients)
                one_body_energies, _, _ = (
                        quad_ham.diagonalizing_bogoliubov_transform())
                params.append(_canonicalize_exponent(
                    -one_body_energies[p]
                    * step_time / numpy.pi, 2))

            # Off-diagonal one-body term
            elif param.letter == 'V':
                p, q, j, _ = param.subscripts
                two_body_coefficients = (
                        self.scaled_density_density_matrices[j])
                params.append(_canonicalize_exponent(
                    -2 * two_body_coefficients[p, q]
                    * interpolation_progress
                    * step_time / numpy.pi, 2))

            # Diagonal two-body terms
            elif param.letter == 'U' and len(param.subscripts) == 3:
                p, j, _ = param.subscripts
                two_body_coefficients = (
                        self.scaled_density_density_matrices[j])
                params.append(_canonicalize_exponent(
                    -two_body_coefficients[p, p]
                    * interpolation_progress
                    * step_time / numpy.pi, 2))

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
