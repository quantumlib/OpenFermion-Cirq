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
from typing import Sequence, Set, Tuple, Union

import numpy

import cirq
from openfermion import (
        QuadraticHamiltonian,
        gaussian_state_preparation_circuit,
        slater_determinant_preparation_circuit)
from openfermion.ops._givens_rotations import (
        fermionic_gaussian_decomposition,
        givens_decomposition_square)

from openfermioncirq import LinearQubit, YXXY


def orbital_basis_change(qubits: Sequence[LinearQubit],
                         transformation_matrix: numpy.ndarray,
                         initial_state: int=None) -> cirq.OP_TREE:
    r"""Perform an orbital basis change.

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


def _occupied_orbitals(computational_basis_state: int) -> Set[int]:
    """Indices of ones in the binary expansion of an integer in little endian
    order. e.g. 10110 -> [1, 2, 4]"""
    bitstring = bin(computational_basis_state)[2:][::-1]
    return {j for j in range(len(bitstring)) if bitstring[j] == '1'}


def _slater_basis_change(qubits: Sequence[LinearQubit],
                         transformation_matrix: numpy.ndarray,
                         initial_state: int=None) -> cirq.OP_TREE:
    n_qubits = transformation_matrix.shape[0]

    if initial_state is None:
        decomposition, _ = givens_decomposition_square(
                transformation_matrix)
        circuit_description = list(reversed(decomposition))
    else:
        occupied_orbitals = _occupied_orbitals(initial_state)
        transformation_matrix = transformation_matrix[list(occupied_orbitals)]
        n_occupied = len(occupied_orbitals)
        # Flip bits so that the first n_occupied are 1 and the rest 0
        yield (cirq.X(qubits[j]) for j in range(n_qubits)
                if (j < n_occupied) != (j in occupied_orbitals))
        circuit_description = slater_determinant_preparation_circuit(
                transformation_matrix)

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _gaussian_basis_change(qubits: Sequence[LinearQubit],
                           transformation_matrix: numpy.ndarray,
                           initial_state: int=None) -> cirq.OP_TREE:
    n_qubits = transformation_matrix.shape[0]

    # Rearrange the transformation matrix because the OpenFermion routine
    # expects it to describe annihilation operators rather than creation
    # operators
    left_block = transformation_matrix[:, :n_qubits]
    right_block = transformation_matrix[:, n_qubits:]
    transformation_matrix = numpy.block(
            [numpy.conjugate(right_block), numpy.conjugate(left_block)])

    decomposition, left_decomposition, _, _ = (
        fermionic_gaussian_decomposition(transformation_matrix))

    if initial_state == 0:
        # Starting with the vacuum state yields additional symmetry
        circuit_description = list(reversed(decomposition))
    else:
        circuit_description = list(reversed(decomposition + left_decomposition))

    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def _ops_from_givens_rotations_circuit_description(
        qubits: Sequence[LinearQubit],
        circuit_description: Sequence[
            Tuple[Union[Tuple[int, int, float, float], str]]]) -> cirq.OP_TREE:
    """Yield operations from a Givens rotations circuit obtained from
    OpenFermion.
    """
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            if op == 'pht':
                yield cirq.X(qubits[-1])
            else:
                i, j, theta, phi = op
                yield YXXY(qubits[i], qubits[j]) ** (theta / numpy.pi)
                yield cirq.Z(qubits[j]) ** (phi / numpy.pi)


def prepare_gaussian_state(qubits: Sequence[LinearQubit],
                           quadratic_hamiltonian: QuadraticHamiltonian,
                           occupied_orbitals: Sequence[int]=None
                           ) -> cirq.OP_TREE:
    """Prepare a fermionic Gaussian state.

    A fermionic Gaussian state is an eigenstate of a quadratic Hamiltonian. If
    the Hamiltonian conserves particle number, then it is a Slater determinant.
    The algorithm used is described in arXiv:1711.05395. It assumes the
    Jordan-Wigner transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        quadratic_hamiltonian: The Hamiltonian whose eigenstate is desired.
        occupied_orbitals: A list of integers representing the indices of the
            pseudoparticle orbitals to occupy in the Gaussian state. The
            orbitals are ordered in ascending order of energy.
            The default behavior is to fill the orbitals with negative energy,
            i.e., prepare the ground state.
    """
    circuit_description, start_orbitals = gaussian_state_preparation_circuit(
            quadratic_hamiltonian, occupied_orbitals)
    for mode in start_orbitals:
        yield cirq.X(qubits[mode])
    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def prepare_slater_determinant(qubits: Sequence[LinearQubit],
                               slater_determinant_matrix: numpy.ndarray
                               ) -> cirq.OP_TREE:
    r"""Prepare a Slater determinant.

    A Slater determinant is described by an :math:`\eta \times N` matrix
    :math:`Q` with orthonormal rows, where :math:`\eta` is the particle number
    and :math:`N` is the total number of modes. The state corresponding to this
    matrix is

    .. math::

        b^\dagger_1 \cdots b^\dagger_{\eta} \lvert \text{vac} \rangle,

    where

    .. math::

        b^\dagger_j = \sum_{k = 1}^N Q_{jk} a^\dagger_k.

    The algorithm used is described in arXiv:1711.05395. It assumes the
    Jordan-Wigner transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        slater_determinant_matrix: The matrix :math:`Q` which describes the
            Slater determinant to be prepared.
    """
    circuit_description = slater_determinant_preparation_circuit(
            slater_determinant_matrix)
    for mode in range(slater_determinant_matrix.shape[0]):
        yield cirq.X(qubits[mode])
    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)


def diagonalizing_basis_change(qubits: Sequence[LinearQubit],
                               quadratic_hamiltonian: QuadraticHamiltonian
                               ) -> cirq.OP_TREE:
    r"""Change to the basis in which a quadratic Hamiltonian is diagonal.

    This circuit performs the transformation to a basis in which the
    Hamiltonian takes the diagonal form

    .. math::

        \sum_{j} \varepsilon_j b^\dagger_j b_j + \text{constant}.

    The :math:`\varepsilon_j` are the energies of the pseudoparticle orbitals
    of the quadratic Hamiltonian, and the resulting basis has the orbitals
    ordered in ascending order of energy.
    The algorithm used is described in arXiv:1711.05395. It assumes the
    Jordan-Wigner transform.

    Args:
        qubits: The qubits to which to apply the circuit.
        quadratic_hamiltonian: The Hamiltonian whose eigenstate is desired.
    """
    circuit_description = quadratic_hamiltonian.diagonalizing_circuit()
    yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description)
