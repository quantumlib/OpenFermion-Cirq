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

from typing import Optional, Sequence, Union
import numpy

import cirq

from openfermion.ops import FermionOperator, InteractionOperator
from openfermion.utils import (get_chemist_two_body_coefficients,
                               low_rank_two_body_decomposition,
                               prepare_one_body_squared_evolution)

from openfermioncirq.ops import XXYYGate
from openfermioncirq.primitives import bogoliubov_transform, swap_network


def low_rank_trotter_step(qubits: Sequence[cirq.QubitId],
                          two_body_operator: Union[FermionOperator,
                                                   InteractionOperator],
                          time: float=1.,
                          truncation_threshold: Optional[float]=None,
                          final_rank: Optional[int]=None,
                          ) -> cirq.OP_TREE:
    """Apply two-body fermionic operator evolution step via low rankness.

    This function is used for perfoming Trotter steps of evolution under a
    Hermitian two-body number-conserving fermionic operator by decomposing
    that operator into singular components and possibly truncating. The
    general idea is based on expressing the two-body operator as
    :math:`\sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s =
    \sum_{j=0}^{J-1} \lambda_j (\sum_{pq} g_{jpq} a^\dagger_p a_q)^2`
    One can then diagonalize the squared one-body component as
    math:`\sum_{pq} g_{pqj} a^\dagger_p a_q =
    R_j (\sum_{p} f_{pj} n_p) R_j^\dagger`
    Then, a 'low rank' Trotter step can be simulated as
    :math:`\prod_{j=0}^{J-1}
    R_j e^{-i \lambda_j \sum_{pq} f_{pj} f_{qj} n_p n_q} R_j^\dagger`.
    One can use the Givens rotation strategy for the :math:`R_j` and one can
    use a swap network to simulate the diagonal :math:`n_p n_q` terms.
    The value of J is either full the square of the number of qubits,
    which would imply no truncation, or it is specified by the user,
    or it is chosen so that
    :math:`\sum_{l=0}^{L-1} (\sum_{pq} |g_{lpq}|)^2 |\lambda_l| < x`
    where x is a truncation threshold specified by user.

    Warnings: due to use of swap networks, if the final rank of the operator
        decomposition is even (which can happen if final_rank is even, or for
        some values of truncation_threshold, or if neither is specified and
        the number of qubits is even, then the order of the qubits in the
        circuit output will be reversed relative to the input.

    TODO:
        Instantiate as a TrotterAlgorithm.
        Add support for a controlled version of this Trotter step.
        Be sure to account for constants in controlled version.
        Add functionality for merging basis transformations.

    Args:
        qubits: The qubits sorted so that the j-th qubit in the Sequence
            represents the j-th qubit or fermionic mode.
        two_body_operator(FermionOperator or InteractionOperator): A real
            number conserving, InteractionOperator or FermionOperator with at
            most two-body interactions.
        time (optional Float): how long to evolve for.
        truncation_threshold (optional Float): the value of x in the expression
            above.
        final_rank (optional int): if provided, this specifies the value of
            J at which to truncate.
    """
    # Get the chemist matrix.
    _, one_body_coefficients, chemist_two_body_coefficients = (
        get_chemist_two_body_coefficients(two_body_operator))
    n_qubits = len(qubits)

    # Perform evolution under the off-diagonal one-body terms.
    yield swap_network(
        qubits, lambda p, q, p_qubit, q_qubit: XXYYGate(
            duration=time * one_body_coefficients[p, q].real).on(
                p_qubit, q_qubit), fermionic=True)
    qubits = qubits[::-1]

    # Perform evolution under the diagonal one-body terms.
    for j in range(n_qubits):
        yield cirq.RotZGate(
            rads=-time * one_body_coefficients[j, j].real).on(qubits[j])

    # Perform the low rank decomposition of two-body operator.
    eigenvalues, one_body_squares, _ = (
        low_rank_two_body_decomposition(
            chemist_two_body_coefficients,
            truncation_threshold=truncation_threshold,
            final_rank=final_rank))

    # Simulate each singular vector.
    for j in range(eigenvalues.size):

        # Get density-density terms and basis transformation matrix.
        density_density_matrix, basis_transformation_matrix = (
            prepare_one_body_squared_evolution(one_body_squares[j]))

        # Simulate basis transformation.
        yield cirq.inverse_of_invertible_op_tree(
            bogoliubov_transform(qubits, basis_transformation_matrix))

        # Perform evolution under the ZZ part of the two-body terms.
        zz_coefficients = time * numpy.real(
            eigenvalues[j] * density_density_matrix)
        yield swap_network(
            qubits, lambda p, q, p_qubit, q_qubit: cirq.Rot11Gate(
                rads=-2. * zz_coefficients[p, q]).on(p_qubit, q_qubit))
        qubits = qubits[::-1]

        # Perform evolution under the local Z part of the two-body terms.
        for k in range(n_qubits):
            yield cirq.RotZGate(rads=-zz_coefficients[k, k]).on(qubits[k])

        # Undo basis transformation.
        yield bogoliubov_transform(qubits, basis_transformation_matrix)
