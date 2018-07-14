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

from typing import Optional, Sequence, Set, Tuple, Union, Iterable, cast

import numpy

import cirq
from openfermion import slater_determinant_preparation_circuit
from openfermion.ops._givens_rotations import (
        fermionic_gaussian_decomposition,
        givens_decomposition_square)

from openfermioncirq import YXXY


def bogoliubov_transform(qubits: Sequence[cirq.QubitId],
                         transformation_matrix: numpy.ndarray,
                         initial_state: Optional[int]=None) -> cirq.OP_TREE:
    r"""Perform a Bogoliubov transformation.

    This circuit performs the transformation to a basis determined by a new set
    of fermionic ladder operators. It performs the unitary :math:`U` such that

    .. math::

        U a^\dagger_p U^{-1} = b^\dagger_p

    where the :math:`a^\dagger_p` are the original creation operators and the
    :math:`b^\dagger_p` are the new creation operators. The new creation
    operators are linear combinations of the original ladder operators with
    coefficients given by the matrix `transformation_matrix`, which will be
    referred to as :math:`W` in the following.

    If :math:`W` is an `N \times N` matrix, then the :math:`b^\dagger_p` are
    given by

    .. math::

        b^\dagger_p = \sum_{q=1}^N W_{pq} a^\dagger_q.

    If :math:`W` is an `N \times 2N` matrix, then the :math:`b^\dagger_p` are
    given by

    .. math::

        b^\dagger_p = \sum_{q=1}^N W_{pq} a^\dagger_q
                      + \sum_{q=N+1}^{2N} W_{pq} a_q.

    This algorithm assumes the Jordan-Wigner Transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        transformation_matrix: The matrix :math:`W` holding the coefficients
            that describe the new creation operators in terms of the original
            ladder operators. Its shape should be either :math:`NxN` or
            :math:`Nx(2N)`, where :math:`N` is the number of qubits.
        initial_state: An optional integer which, if specified, will cause this
            function to assume that the given qubits are in the computational
            basis state corresponding to this integer. This assumption enables
            optimizations that result in a circuit with fewer gates.
            Integers are mapped to computational basis states via "big endian"
            ordering of the binary representation of the integer. For example,
            The computational basis state on five qubits with the first and
            second qubits set to one is 0b11000, which has the integer value 24.
    """
    n_qubits = len(qubits)
    shape = transformation_matrix.shape

    if shape == (n_qubits, n_qubits):
        # We're performing a particle-number conserving "Slater" basis change
        yield _slater_basis_change(qubits,
                                   transformation_matrix,
                                   initial_state=initial_state)
    elif shape == (n_qubits, 2 * n_qubits):
        # We're performing a more general Gaussian unitary
        yield _gaussian_basis_change(qubits,
                                     transformation_matrix,
                                     initial_state=initial_state)
    else:
        raise ValueError('Bad shape for transformation_matrix. '
                         'Expected {} or {} but got {}.'.format(
                             (n_qubits, n_qubits),
                             (n_qubits, 2 * n_qubits),
                             shape))


def _occupied_orbitals(computational_basis_state: int, n_qubits) -> Set[int]:
    """Indices of ones in the binary expansion of an integer in big endian
    order. e.g. 010110 -> [1, 3, 4]"""
    bitstring = format(computational_basis_state, 'b').zfill(n_qubits)
    return {j for j in range(len(bitstring)) if bitstring[j] == '1'}


def _slater_basis_change(qubits: Sequence[cirq.QubitId],
                         transformation_matrix: numpy.ndarray,
                         initial_state: int=None) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    if initial_state is None:
        decomposition, diagonal = givens_decomposition_square(
                transformation_matrix)
        circuit_description = list(reversed(decomposition))
        # The initial state is not a computational basis state so the
        # phases left on the diagonal in the decomposition matter
        yield (cirq.RotZGate(rads=numpy.angle(diagonal[j])).on(qubits[j])
               for j in range(n_qubits))
    else:
        occupied_orbitals = _occupied_orbitals(initial_state, n_qubits)
        transformation_matrix = transformation_matrix[list(occupied_orbitals)]
        n_occupied = len(occupied_orbitals)
        # Flip bits so that the first n_occupied are 1 and the rest 0
        yield (cirq.X(qubits[j]) for j in range(n_qubits)
               if (j < n_occupied) != (j in occupied_orbitals))
        circuit_description = slater_determinant_preparation_circuit(
                transformation_matrix)

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _gaussian_basis_change(qubits: Sequence[cirq.QubitId],
                           transformation_matrix: numpy.ndarray,
                           initial_state: int=None) -> cirq.OP_TREE:
    n_qubits = len(qubits)

    # Rearrange the transformation matrix because the OpenFermion routine
    # expects it to describe annihilation operators rather than creation
    # operators
    left_block = transformation_matrix[:, :n_qubits]
    right_block = transformation_matrix[:, n_qubits:]
    transformation_matrix = numpy.block(
            [numpy.conjugate(right_block), numpy.conjugate(left_block)])

    decomposition, left_decomposition, _, left_diagonal = (
        fermionic_gaussian_decomposition(transformation_matrix))

    if initial_state == 0:
        # Starting with the vacuum state yields additional symmetry
        circuit_description = list(reversed(decomposition))
    else:
        if initial_state is None:
            # The initial state is not a computational basis state so the
            # phases left on the diagonal in the Givens decomposition matter
            yield (cirq.RotZGate(rads=
                       numpy.angle(left_diagonal[j])).on(qubits[j])
                   for j in range(n_qubits))
        circuit_description = list(reversed(decomposition + left_decomposition))

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _ops_from_givens_rotations_circuit_description(
        qubits: Sequence[cirq.QubitId],
        circuit_description: Iterable[Iterable[
            Union[str, Tuple[int, int, float, float]]]]) -> cirq.OP_TREE:
    """Yield operations from a Givens rotations circuit obtained from
    OpenFermion.
    """
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            if op == 'pht':
                yield cirq.X(qubits[-1])
            else:
                i, j, theta, phi = cast(Tuple[int, int, float, float], op)
                yield YXXY(qubits[i], qubits[j]) ** (2 * theta / numpy.pi)
                yield cirq.Z(qubits[j]) ** (phi / numpy.pi)
