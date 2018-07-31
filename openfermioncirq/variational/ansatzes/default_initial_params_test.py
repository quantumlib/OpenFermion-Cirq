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
import openfermion
from openfermion.utils._testing_utils import (
        random_diagonal_coulomb_hamiltonian)

from openfermioncirq import (
        HamiltonianObjective,
        SplitOperatorTrotterAnsatz,
        SwapNetworkTrotterAnsatz,
        VariationalStudy,
        prepare_gaussian_state,
        simulate_trotter)
from openfermioncirq.trotter import LINEAR_SWAP_NETWORK, SPLIT_OPERATOR


# Construct a Hamiltonian for testing
test_hamiltonian = random_diagonal_coulomb_hamiltonian(4, real=True, seed=47141)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, test_hamiltonian, 5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, test_hamiltonian, 5e-5),
])
def test_trotter_ansatzes_default_initial_params_iterations_1(
        ansatz_factory, trotter_algorithm, hamiltonian, atol):
    """Check that a Trotter ansatz with one iteration and default parameters
    is consistent with time evolution with one Trotter step."""

    ansatz = ansatz_factory(hamiltonian, iterations=1)
    objective = HamiltonianObjective(hamiltonian)

    qubits = ansatz.qubits

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz circuit and objective
    result = simulator.simulate(
            preparation_circuit + ansatz.circuit,
            param_resolver=
                ansatz.param_resolver(ansatz.default_initial_params()),
            qubit_order=ansatz.qubit_permutation(qubits)
    )
    obj_val = objective.value(result)

    # Compute value using study
    study = VariationalStudy(
            'study',
            ansatz,
            objective,
            preparation_circuit=preparation_circuit)
    study_val = study.value_of(ansatz.default_initial_params())

    # Compute value by simulating time evolution
    half_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
            one_body=hamiltonian.one_body,
            two_body=0.5 * hamiltonian.two_body)
    simulation_circuit = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                half_way_hamiltonian,
                time=ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=1,
                algorithm=trotter_algorithm)
    )
    result = simulator.simulate(preparation_circuit + simulation_circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            objective._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(obj_val, study_val, atol=atol)
    numpy.testing.assert_allclose(obj_val, correct_val, atol=atol)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, test_hamiltonian, 5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, test_hamiltonian, 5e-5),
])
def test_trotter_ansatzes_default_initial_params_iterations_2(
        ansatz_factory, trotter_algorithm, hamiltonian, atol):
    """Check that a Trotter ansatz with two iterations and default parameters
    is consistent with time evolution with two Trotter steps."""

    ansatz = ansatz_factory(hamiltonian, iterations=2)
    objective = HamiltonianObjective(hamiltonian)

    qubits = ansatz.qubits

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz circuit and objective
    result = simulator.simulate(
            preparation_circuit + ansatz.circuit,
            param_resolver=
                ansatz.param_resolver(ansatz.default_initial_params()),
            qubit_order=ansatz.qubit_permutation(qubits)
    )
    obj_val = objective.value(result)

    # Compute value using study
    study = VariationalStudy(
            'study',
            ansatz,
            objective,
            preparation_circuit=preparation_circuit)
    study_val = study.value_of(ansatz.default_initial_params())

    # Compute value by simulating time evolution
    quarter_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
            one_body=hamiltonian.one_body,
            two_body=0.25 * hamiltonian.two_body)
    three_quarters_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
            one_body=hamiltonian.one_body,
            two_body=0.75 * hamiltonian.two_body)
    simulation_circuit = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                quarter_way_hamiltonian,
                time=0.5 * ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=1,
                algorithm=trotter_algorithm),
            simulate_trotter(
                qubits,
                three_quarters_way_hamiltonian,
                time=0.5 * ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=1,
                algorithm=trotter_algorithm)
    )
    result = simulator.simulate(preparation_circuit + simulation_circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            objective._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(obj_val, study_val, atol=atol)
    numpy.testing.assert_allclose(obj_val, correct_val, atol=atol)
