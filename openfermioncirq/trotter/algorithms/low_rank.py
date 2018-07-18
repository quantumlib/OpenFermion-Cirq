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

"""A Trotter algorithm using the low rank decomposition strategy."""

from typing import Optional, Sequence, TYPE_CHECKING, Tuple

import numpy

import cirq
from openfermion.ops import InteractionOperator
from openfermion.utils import (get_chemist_two_body_coefficients,
                               low_rank_two_body_decomposition,
                               prepare_one_body_squared_evolution)

from openfermioncirq import (
        ControlledXXYYGate,
        Rot111Gate,
        XXYYGate,
        bogoliubov_transform,
        swap_network)
from openfermioncirq.trotter.trotter_algorithm import (
        Hamiltonian,
        TrotterStep,
        TrotterAlgorithm)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


class LowRankTrotterAlgorithm(TrotterAlgorithm):
    """A Trotter algorithm using the low rank decomposition strategy.

    This algorithm simulates an InteractionOperator with real coefficients.
    The one-body terms are simulated using a fermionic swap network.
    To simulate the two-body terms, the two-body tensor is decomposed into
    singular components and possibly truncating. Then, each singular component
    is simulated in the appropriate basis using a (non-fermionic) swap network.
    The general idea is based on expressing the two-body operator as
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s =
    \sum_{j=0}^{J-1} \lambda_j (\sum_{pq} g_{jpq} a^\dagger_p a_q)^2`
    One can then diagonalize the squared one-body component as
    math:`\sum_{pq} g_{pqj} a^\dagger_p a_q =
    R_j (\sum_{p} f_{pj} n_p) R_j^\dagger`
    Then, a 'low rank' Trotter step of the two-body tensor can be simulated as
    :math:`\prod_{j=0}^{J-1}
    R_j e^{-i \lambda_j \sum_{pq} f_{pj} f_{qj} n_p n_q} R_j^\dagger`.
    One can use the Givens rotation strategy for the :math:`R_j` and one can
    use a swap network to simulate the diagonal :math:`n_p n_q` terms.
    The value of J is either fully the square of the number of qubits,
    which would imply no truncation, or it is specified by the user,
    or it is chosen so that
    :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`
    where x is a truncation threshold specified by user.
    """

    supported_types = {InteractionOperator}

    def __init__(self,
                 truncation_threshold: Optional[float]=None,
                 final_rank: Optional[int]=None) -> None:
        """
        Args:
            truncation_threshold: The value of x from the docstring of
                this class.
            final_rank: If provided, this specifies the value of J at which to
                truncate.
        """
        self.truncation_threshold = truncation_threshold
        self.final_rank = final_rank

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return AsymmetricLowRankTrotterStep(
                hamiltonian, self.truncation_threshold, self.final_rank)

    def controlled_asymmetric(self, hamiltonian: Hamiltonian
                              ) -> Optional[TrotterStep]:
        return ControlledAsymmetricLowRankTrotterStep(
                hamiltonian, self.truncation_threshold, self.final_rank)


LOW_RANK = LowRankTrotterAlgorithm()


class LowRankTrotterStep(TrotterStep):

    def __init__(self,
                 hamiltonian: InteractionOperator,
                 truncation_threshold: Optional[float]=None,
                 final_rank: Optional[int]=None) -> None:

        self.truncation_threshold = truncation_threshold
        self.final_rank = final_rank

        # Get the chemist matrix.
        (self.constant,
         self.one_body_coefficients,
         chemist_two_body_coefficients) = (
                 get_chemist_two_body_coefficients(hamiltonian))

        # Perform the low rank decomposition of two-body operator.
        self.eigenvalues, self.one_body_squares, _ = (
            low_rank_two_body_decomposition(
                chemist_two_body_coefficients,
                truncation_threshold=self.truncation_threshold,
                final_rank=self.final_rank))

        # Get scaled density-density terms and basis transformation matrices.
        self.scaled_density_density_matrices = []  # type: List[numpy.ndarray]
        self.basis_change_matrices = []            # type: List[numpy.ndarray]
        for j in range(len(self.eigenvalues)):
            density_density_matrix, basis_change_matrix = (
                prepare_one_body_squared_evolution(self.one_body_squares[j]))
            self.scaled_density_density_matrices.append(
                    numpy.real(self.eigenvalues[j] * density_density_matrix))
            self.basis_change_matrices.append(basis_change_matrix)

        super().__init__(hamiltonian)


class AsymmetricLowRankTrotterStep(LowRankTrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Simulate the off-diagonal one-body terms.
        yield swap_network(
                qubits,
                lambda p, q, a, b: XXYYGate(duration=
                    self.one_body_coefficients[p, q].real * time).on(a, b),
                fermionic=True)
        qubits = qubits[::-1]

        # Simulate the diagonal one-body terms.
        yield (cirq.RotZGate(rads=
                   -self.one_body_coefficients[j, j].real * time).on(qubits[j])
               for j in range(n_qubits))

        # Simulate each singular vector of the two-body terms.
        prior_basis_matrix = numpy.identity(n_qubits, complex)
        for j in range(len(self.eigenvalues)):

            # Get the basis change matrix and its inverse.
            two_body_coefficients = self.scaled_density_density_matrices[j]
            basis_change_matrix = self.basis_change_matrices[j]

            # If the basis change matrix is unitary, merge rotations.
            # Otherwise, you must simulate two basis rotations.
            is_unitary = cirq.is_unitary(basis_change_matrix)
            if is_unitary:
                inverse_basis_matrix = numpy.conjugate(
                    numpy.transpose(basis_change_matrix))
                merged_basis_matrix = numpy.dot(prior_basis_matrix,
                                                inverse_basis_matrix)
                yield bogoliubov_transform(qubits, merged_basis_matrix)
            else:
                # TODO add LiH test to cover this.
                # coverage: ignore
                if j:
                    yield bogoliubov_transform(qubits, prior_basis_matrix)
                yield cirq.inverse(
                    bogoliubov_transform(qubits, basis_change_matrix))

            # Simulate the off-diagonal two-body terms.
            yield swap_network(
                    qubits,
                    lambda p, q, a, b: cirq.Rot11Gate(rads=
                        -2 * two_body_coefficients[p, q] * time).on(a, b))
            qubits = qubits[::-1]

            # Simulate the diagonal two-body terms.
            yield (cirq.RotZGate(rads=
                       -two_body_coefficients[k, k] * time).on(qubits[k])
                   for k in range(n_qubits))

            # Undo basis transformation non-unitary case.
            # Else, set basis change matrix to prior matrix for merging.
            if is_unitary:
                prior_basis_matrix = basis_change_matrix
            else:
                # TODO add LiH test to cover this.
                # coverage: ignore
                yield bogoliubov_transform(qubits, basis_change_matrix)
                prior_basis_matrix = numpy.identity(n_qubits, complex)

        # Undo final basis transformation in unitary case.
        if is_unitary:
            yield bogoliubov_transform(qubits, prior_basis_matrix)

    def step_qubit_permutation(self,
                               qubits: Sequence[cirq.QubitId],
                               control_qubit: Optional[cirq.QubitId]=None
                               ) -> Tuple[Sequence[cirq.QubitId],
                                          Optional[cirq.QubitId]]:
        # A Trotter step reverses the qubit ordering when the number of
        # eigenvalues is even
        if len(self.eigenvalues) & 1:
            return qubits, None
        else:
            return qubits[::-1], None

    def finish(self,
               qubits: Sequence[cirq.QubitId],
               n_steps: int,
               control_qubit: Optional[cirq.QubitId]=None,
               omit_final_swaps: bool=False
               ) -> cirq.OP_TREE:
        if not omit_final_swaps:
            # If the number of fermionic swap networks was odd,
            # swap the modes back
            if n_steps & 1:
                yield swap_network(qubits, fermionic=True)
                # If the total number of swap networks was odd,
                # swap the qubits back
                if len(self.eigenvalues) & 1:
                    yield swap_network(qubits)


class ControlledAsymmetricLowRankTrotterStep(LowRankTrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Simulate the off-diagonal one-body terms.
        yield swap_network(
                qubits,
                lambda p, q, a, b: ControlledXXYYGate(duration=
                    self.one_body_coefficients[p, q].real * time).on(
                        control_qubit, a, b),
                fermionic=True)
        qubits = qubits[::-1]

        # Simulate the diagonal one-body terms.
        yield (cirq.Rot11Gate(rads=
                   -self.one_body_coefficients[j, j].real * time).on(
                       control_qubit, qubits[j])
               for j in range(n_qubits))

        # Simulate each singular vector of the two-body terms.
        prior_basis_matrix = numpy.identity(n_qubits, complex)
        for j in range(len(self.eigenvalues)):

            # Get the basis change matrix and its inverse.
            two_body_coefficients = self.scaled_density_density_matrices[j]
            basis_change_matrix = self.basis_change_matrices[j]

            # If the basis change matrix is unitary, merge rotations.
            # Otherwise, you must simulate two basis rotations.
            is_unitary = cirq.is_unitary(basis_change_matrix)
            if is_unitary:
                inverse_basis_matrix = numpy.conjugate(
                    numpy.transpose(basis_change_matrix))
                merged_basis_matrix = numpy.dot(prior_basis_matrix,
                                                inverse_basis_matrix)
                yield bogoliubov_transform(qubits, merged_basis_matrix)
            else:
                # TODO add LiH test to cover this.
                # coverage: ignore
                if j:
                    yield bogoliubov_transform(qubits, prior_basis_matrix)
                yield cirq.inverse(
                    bogoliubov_transform(qubits, basis_change_matrix))

            # Simulate the off-diagonal two-body terms.
            yield swap_network(
                    qubits,
                    lambda p, q, a, b: Rot111Gate(rads=
                        -2 * two_body_coefficients[p, q] * time).on(
                            control_qubit, a, b))
            qubits = qubits[::-1]

            # Simulate the diagonal two-body terms.
            yield (cirq.Rot11Gate(rads=
                       -two_body_coefficients[k, k] * time).on(
                           control_qubit, qubits[k])
                   for k in range(n_qubits))

            # Undo basis transformation non-unitary case.
            # Else, set basis change matrix to prior matrix for merging.
            if is_unitary:
                prior_basis_matrix = basis_change_matrix
            else:
                # TODO add LiH test to cover this.
                # coverage: ignore
                yield bogoliubov_transform(qubits, basis_change_matrix)
                prior_basis_matrix = numpy.identity(n_qubits, complex)

        # Undo final basis transformation in unitary case.
        if is_unitary:
            yield bogoliubov_transform(qubits, prior_basis_matrix)

        # Apply phase from constant term
        yield cirq.RotZGate(rads=-self.constant * time).on(control_qubit)

    def step_qubit_permutation(self,
                               qubits: Sequence[cirq.QubitId],
                               control_qubit: Optional[cirq.QubitId]=None
                               ) -> Tuple[Sequence[cirq.QubitId],
                                          Optional[cirq.QubitId]]:
        # A Trotter step reverses the qubit ordering when the number of
        # eigenvalues is even
        if len(self.eigenvalues) & 1:
            return qubits, control_qubit
        else:
            return qubits[::-1], control_qubit

    def finish(self,
               qubits: Sequence[cirq.QubitId],
               n_steps: int,
               control_qubit: Optional[cirq.QubitId]=None,
               omit_final_swaps: bool=False
               ) -> cirq.OP_TREE:
        if not omit_final_swaps:
            # If the number of fermionic swap networks was odd,
            # swap the modes back
            if n_steps & 1:
                yield swap_network(qubits, fermionic=True)
                # If the total number of swap networks was odd,
                # swap the qubits back
                if len(self.eigenvalues) & 1:
                    yield swap_network(qubits)
