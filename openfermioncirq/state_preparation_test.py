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
from cirq import LineQubit
from openfermion import get_sparse_operator
from openfermion.utils._testing_utils import random_quadratic_hamiltonian

from openfermioncirq.state_preparation import (
        bogoliubov_transform,
        prepare_gaussian_state,
        prepare_slater_determinant)


def fourier_transform_matrix(n_modes):
    root_of_unity = numpy.exp(2j * numpy.pi / n_modes)
    return numpy.array([[root_of_unity ** (j * k) for k in range(n_modes)]
                        for j in range(n_modes)])


@pytest.mark.parametrize(
        'transformation_matrix, initial_state, correct_state',
        [(fourier_transform_matrix(3), 4, numpy.array(
            [0, 1, 1, 0, 1, 0, 0, 0]) / numpy.sqrt(3)),
         (fourier_transform_matrix(3), 3, numpy.array(
            [0, 0, 0, numpy.exp(2j * numpy.pi / 3) - 1,
                0, 1 - numpy.exp(2j * numpy.pi / 3),
                numpy.exp(2j * numpy.pi / 3) - 1, 0]) / 3),
        ])
def test_bogoliubov_transform_fourier_transform(transformation_matrix,
                                                initial_state,
                                                correct_state,
                                                atol=1e-7):
    simulator = cirq.google.XmonSimulator()
    n_qubits = transformation_matrix.shape[0]
    qubits = LineQubit.range(n_qubits)

    circuit = cirq.Circuit.from_ops(bogoliubov_transform(
        qubits, transformation_matrix, initial_state=initial_state))
    result = simulator.simulate(circuit, initial_state=initial_state)
    state = result.final_state

    assert cirq.allclose_up_to_global_phase(state, correct_state, atol=atol)


@pytest.mark.parametrize(
        'n_qubits, conserves_particle_number',
        [(4, True), (4, False), (5, True), (5, False)])
def test_bogoliubov_transform_quadratic_hamiltonian(n_qubits,
                                                    conserves_particle_number,
                                                    atol=1e-5):
    simulator = cirq.google.XmonSimulator()
    qubits = LineQubit.range(n_qubits)

    # Initialize a random quadratic Hamiltonian
    quad_ham = random_quadratic_hamiltonian(
            n_qubits, conserves_particle_number, real=False)
    quad_ham_sparse = get_sparse_operator(quad_ham)

    # Compute the orbital energies and circuit
    orbital_energies, constant = quad_ham.orbital_energies()
    transformation_matrix = quad_ham.diagonalizing_bogoliubov_transform()
    circuit = cirq.Circuit.from_ops(
            bogoliubov_transform(qubits, transformation_matrix))

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

        # Construct initial state
        initial_state = sum(1 << (n_qubits - 1 - int(i))
                            for i in occupied_orbitals)

        # Get the state using a circuit simulation
        result = simulator.simulate(circuit,
                                    qubit_order=qubits,
                                    initial_state=initial_state)
        state1 = result.final_state

        # Also test the option to start with a computational basis state
        special_circuit = cirq.Circuit.from_ops(bogoliubov_transform(
            qubits,
            transformation_matrix,
            initial_state=initial_state))
        result = simulator.simulate(special_circuit,
                                    qubit_order=qubits,
                                    initial_state=initial_state)
        state2 = result.final_state

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
    simulator = cirq.google.XmonSimulator()
    qubits = LineQubit.range(n_qubits)

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
    result = simulator.simulate(circuit, qubit_order=qubits)
    state = result.final_state

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
def test_prepare_slater_determinant(slater_determinant_matrix,
                                    correct_state,
                                    atol=1e-7):
    simulator = cirq.google.XmonSimulator()
    n_qubits = slater_determinant_matrix.shape[1]
    qubits = LineQubit.range(n_qubits)

    circuit = cirq.Circuit.from_ops(
            prepare_slater_determinant(qubits, slater_determinant_matrix))
    result = simulator.simulate(circuit, qubit_order=qubits)
    state = result.final_state

    assert cirq.allclose_up_to_global_phase(state, correct_state, atol=atol)
