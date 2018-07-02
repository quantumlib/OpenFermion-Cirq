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

from typing import Sequence, Tuple, Union, Iterable, cast

import numpy

import cirq
from openfermion import (
        QuadraticHamiltonian,
        gaussian_state_preparation_circuit,
        slater_determinant_preparation_circuit)

from openfermioncirq import YXXY


def prepare_gaussian_state(qubits: Sequence[cirq.QubitId],
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


def prepare_slater_determinant(qubits: Sequence[cirq.QubitId],
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


def _ops_from_givens_rotations_circuit_description(
        qubits: Sequence[cirq.QubitId],
        circuit_description: Iterable[Iterable[
            Union[str, Tuple[int, int, float, float]]]]
) -> cirq.OP_TREE:
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
