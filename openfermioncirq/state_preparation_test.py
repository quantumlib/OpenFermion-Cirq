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
import numpy
import pytest

import cirq
from openfermion import get_sparse_operator
from openfermion.utils._testing_utils import random_quadratic_hamiltonian

from openfermioncirq import LinearQubit

from openfermioncirq.state_preparation import (
        diagonalizing_basis_change,
        orbital_basis_change,
        prepare_gaussian_state,
        prepare_slater_determinant)


def fourier_transform_matrix(n_modes):
    root_of_unity = numpy.exp(2j * numpy.pi / n_modes)
    return numpy.array([[root_of_unity ** (j * k) for k in range(n_modes)]
                        for j in range(n_modes)])


@pytest.mark.parametrize(
        'transformation_matrix, initial_state, correct_state',
        [(fourier_transform_matrix(3), 4, numpy.array(
            [0, 1, numpy.exp(4j * numpy.pi / 3), 0,
                numpy.exp(2j * numpy.pi / 3), 0, 0, 0]) / numpy.sqrt(3)),
         (fourier_transform_matrix(3), 3, numpy.array(
            [0, 0, 0, 1, 0, 1 + numpy.exp(2j * numpy.pi / 3),
                numpy.exp(2j * numpy.pi / 3), 0]) / numpy.sqrt(3)),
        ])
def test_orbital_basis_change_fourier_transform_test(transformation_matrix,
                                                     initial_state,
                                                     correct_state,
                                                     atol=1e-6):
    simulator = cirq.google.Simulator()
    n_qubits = transformation_matrix.shape[0]
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    circuit = cirq.Circuit.from_ops(orbital_basis_change(
        qubits, transformation_matrix, initial_state=initial_state))
    result = simulator.run(circuit,
                           initial_state=initial_state)
    state = result.final_states[0]

    assert cirq.allclose_up_to_global_phase(state, correct_state, atol=atol)


@pytest.mark.parametrize(
        'n_qubits, conserves_particle_number',
        [(4, True), (4, False), (5, True), (5, False)])
def test_orbital_basis_change_quadratic_hamiltonian(n_qubits,
                                                    conserves_particle_number,
                                                    atol=1e-5):
    simulator = cirq.google.Simulator()
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
            n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the orbital energies and circuit
    orbital_energies, constant = quad_ham.orbital_energies()
    transformation_matrix = quad_ham.diagonalizing_bogoliubov_transform()
    circuit = cirq.Circuit.from_ops(
            orbital_basis_change(qubits, transformation_matrix))

    # Pick some random eigenstates to prepare, which correspond to random
    # subsets of [0 ... n_qubits - 1]
    n_eigenstates = min(1 << n_qubits, 5)
    subsets = [numpy.random.choice(range(n_qubits),
                                   numpy.random.randint(1, n_qubits + 1),
                                   False)
               for _ in range(n_eigenstates)]

    for occupied_orbitals in subsets:
        # Compute the energy of this eigenstate
        energy = (sum(orbital_energies[i] for i in occupied_orbitals) +
                  constant)

        # Get the state using a circuit simulation
        result = simulator.run(
                circuit,
                qubit_order=qubits[::-1],
                initial_state=sum(1 << (n_qubits - 1 - int(i))
                                  for i in occupied_orbitals))
        state1 = result.final_states[0]

        # Also test the option to start with a computational basis state
        special_circuit = cirq.Circuit.from_ops(orbital_basis_change(
            qubits,
            transformation_matrix,
            initial_state=sum(1 << int(i) for i in occupied_orbitals)))
        result = simulator.run(
                special_circuit,
                qubit_order=qubits[::-1],
                initial_state=sum(1 << (n_qubits - 1 - int(i))
                                  for i in occupied_orbitals))
        state2 = result.final_states[0]

        # Check that the result is an eigenstate with the correct eigenvalue
        numpy.testing.assert_allclose(
                quad_ham_sparse.dot(state1), energy * state1, atol=atol)
        numpy.testing.assert_allclose(
                quad_ham_sparse.dot(state2), energy * state2, atol=atol)


@pytest.mark.parametrize(
        'n_qubits, conserves_particle_number, occupied_orbitals',
        [(4, True, None),
         (4, False, None),
         (5, True, None),
         (5, False, None),
         (5, True, range(4)),
         (5, False, (0, 2, 3))])
def test_prepare_gaussian_state(n_qubits,
                                conserves_particle_number,
                                occupied_orbitals,
                                atol=1e-5):
    simulator = cirq.google.Simulator()
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
            n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the energy of the desired state
    if occupied_orbitals is None:
        energy = quad_ham.ground_energy()
    else:
        orbital_energies, constant = quad_ham.orbital_energies()
        energy = sum(orbital_energies[i] for i in occupied_orbitals) + constant

    # Get the state using a circuit simulation
    circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(qubits, quad_ham, occupied_orbitals))
    result = simulator.run(circuit, qubit_order=qubits[::-1])
    state = result.final_states[0]

    # Check that the result is an eigenstate with the correct eigenvalue
    numpy.testing.assert_allclose(
            quad_ham_sparse.dot(state), energy * state, atol=atol)


@pytest.mark.parametrize(
        'slater_determinant_matrix, correct_state',
        [(numpy.array([[1, 1]]) / numpy.sqrt(2),
          numpy.array([0, 1, 1, 0]) / numpy.sqrt(2)),

         (numpy.array([[1, 1j]]) / numpy.sqrt(2),
          numpy.array([0, 1j, 1, 0]) / numpy.sqrt(2)),

         (numpy.array([[1, 1, 1], [1, numpy.exp(2j * numpy.pi / 3),
             numpy.exp(4j * numpy.pi / 3)]]) / numpy.sqrt(3),
          numpy.array([0, 0, 0, numpy.exp(2j * numpy.pi / 3), 0,
             1 + numpy.exp(2j * numpy.pi / 3), 1, 0]) / numpy.sqrt(3))
        ])
def test_prepare_slater_determinant_test(slater_determinant_matrix,
                                         correct_state,
                                         atol=1e-7):
    simulator = cirq.google.Simulator()
    n_qubits = slater_determinant_matrix.shape[1]
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    circuit = cirq.Circuit.from_ops(
            prepare_slater_determinant(qubits, slater_determinant_matrix))
    result = simulator.run(circuit, qubit_order=qubits[::-1])
    state = result.final_states[0]

    assert cirq.allclose_up_to_global_phase(state, correct_state, atol=atol)


@pytest.mark.parametrize(
        'n_qubits, conserves_particle_number',
        [(3, True), (3, False)])
def test_diagonalizing_basis_change(n_qubits,
                                    conserves_particle_number,
                                    atol=1e-5):
    simulator = cirq.google.Simulator()
    qubits = [LinearQubit(i) for i in range(n_qubits)]

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
            n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the orbital energies and diagonalizing circuit
    orbital_energies, constant = quad_ham.orbital_energies()
    circuit = cirq.Circuit.from_ops(
            diagonalizing_basis_change(qubits, quad_ham))

    # Pick some random eigenstates to prepare, which correspond to random
    # subsets of [0 ... n_qubits - 1]
    n_eigenstates = min(1 << n_qubits, 5)
    subsets = [numpy.random.choice(range(n_qubits),
                                   numpy.random.randint(1, n_qubits + 1),
                                   False)
               for _ in range(n_eigenstates)]

    for occupied_orbitals in subsets:
        # Compute the energy of this eigenstate
        energy = (sum(orbital_energies[i] for i in occupied_orbitals) +
                  constant)

        # Get the state using a circuit simulation
        result = simulator.run(
                circuit,
                qubit_order=qubits[::-1],
                initial_state=sum(1 << (n_qubits - 1 - int(i))
                                  for i in occupied_orbitals))
        state = result.final_states[0]

        # Check that the result is an eigenstate with the correct eigenvalue
        numpy.testing.assert_allclose(
                quad_ham_sparse.dot(state), energy * state, atol=atol)
