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

from openfermioncirq import (
        HamiltonianObjective,
        LowRankTrotterAnsatz,
        SplitOperatorTrotterAnsatz,
        SwapNetworkTrotterAnsatz,
        VariationalStudy,
        prepare_gaussian_state,
        simulate_trotter)
from openfermioncirq.trotter import (
        LINEAR_SWAP_NETWORK, LOW_RANK, LowRankTrotterAlgorithm, SPLIT_OPERATOR)


# 4-qubit random DiagonalCoulombHamiltonian
diag_coul_hamiltonian = openfermion.random_diagonal_coulomb_hamiltonian(
        4, real=True, seed=47141)

# 4-qubit H2 2-2 with bond length 0.7414
bond_length = 0.7414
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
h2_hamiltonian = openfermion.load_molecular_hamiltonian(
        geometry, 'sto-3g', 1, format(bond_length), 2, 2)

# 4-qubit LiH 2-2 with bond length 1.45
bond_length = 1.45
geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., bond_length))]
lih_hamiltonian = openfermion.load_molecular_hamiltonian(
        geometry, 'sto-3g', 1, format(bond_length), 2, 2)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, order, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, 1, diag_coul_hamiltonian,
        5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, 1, diag_coul_hamiltonian,
        5e-5),
    (LowRankTrotterAnsatz, LOW_RANK, 0, h2_hamiltonian, 5e-5),
    (lambda hamiltonian, iterations: LowRankTrotterAnsatz(
            hamiltonian,
            iterations=iterations,
            final_rank=3),
        LowRankTrotterAlgorithm(final_rank=3),
        0, lih_hamiltonian, 5e-5),
])
def test_trotter_ansatzes_default_initial_params_iterations_1(
        ansatz_factory, trotter_algorithm, order, hamiltonian, atol):
    """Check that a Trotter ansatz with one iteration and default parameters
    is consistent with time evolution with one Trotter step."""

    ansatz = ansatz_factory(hamiltonian, iterations=1)
    objective = HamiltonianObjective(hamiltonian)

    qubits = ansatz.qubits

    if isinstance(hamiltonian, openfermion.DiagonalCoulombHamiltonian):
        one_body = hamiltonian.one_body
    elif isinstance(hamiltonian, openfermion.InteractionOperator):
        one_body = hamiltonian.one_body_tensor

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )

    # Compute value using ansatz circuit and objective
    circuit = cirq.resolve_parameters(
            preparation_circuit + ansatz.circuit,
            ansatz.param_resolver(ansatz.default_initial_params()))
    result = circuit.apply_unitary_effect_to_state(
            qubit_order=ansatz.qubit_permutation(qubits))
    obj_val = objective.value(result)

    # Compute value using study
    study = VariationalStudy(
            'study',
            ansatz,
            objective,
            preparation_circuit=preparation_circuit)
    study_val = study.value_of(ansatz.default_initial_params())

    # Compute value by simulating time evolution
    if isinstance(hamiltonian, openfermion.DiagonalCoulombHamiltonian):
        half_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
                one_body=hamiltonian.one_body,
                two_body=0.5 * hamiltonian.two_body)
    elif isinstance(hamiltonian, openfermion.InteractionOperator):
        half_way_hamiltonian = openfermion.InteractionOperator(
                constant=hamiltonian.constant,
                one_body_tensor=hamiltonian.one_body_tensor,
                two_body_tensor=0.5 * hamiltonian.two_body_tensor)

    simulation_circuit = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                half_way_hamiltonian,
                time=ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=order,
                algorithm=trotter_algorithm)
    )
    final_state = (preparation_circuit + simulation_circuit
                  ).apply_unitary_effect_to_state()
    correct_val = openfermion.expectation(
            objective._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(obj_val, study_val, atol=atol)
    numpy.testing.assert_allclose(obj_val, correct_val, atol=atol)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, order, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, 1, diag_coul_hamiltonian,
        5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, 1, diag_coul_hamiltonian,
        5e-5),
    (LowRankTrotterAnsatz, LOW_RANK, 0, h2_hamiltonian, 5e-5),
    (lambda hamiltonian, iterations: LowRankTrotterAnsatz(
            hamiltonian,
            iterations=iterations,
            final_rank=3),
        LowRankTrotterAlgorithm(final_rank=3),
        0, lih_hamiltonian, 1e-3),
])
def test_trotter_ansatzes_default_initial_params_iterations_2(
        ansatz_factory, trotter_algorithm, order, hamiltonian, atol):
    """Check that a Trotter ansatz with two iterations and default parameters
    is consistent with time evolution with two Trotter steps."""

    ansatz = ansatz_factory(hamiltonian, iterations=2)
    objective = HamiltonianObjective(hamiltonian)

    qubits = ansatz.qubits

    if isinstance(hamiltonian, openfermion.DiagonalCoulombHamiltonian):
        one_body = hamiltonian.one_body
    elif isinstance(hamiltonian, openfermion.InteractionOperator):
        one_body = hamiltonian.one_body_tensor

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )

    # Compute value using ansatz circuit and objective
    circuit = cirq.resolve_parameters(
            preparation_circuit + ansatz.circuit,
            ansatz.param_resolver(ansatz.default_initial_params()))
    result = circuit.apply_unitary_effect_to_state(
            qubit_order=ansatz.qubit_permutation(qubits))
    obj_val = objective.value(result)

    # Compute value using study
    study = VariationalStudy(
            'study',
            ansatz,
            objective,
            preparation_circuit=preparation_circuit)
    study_val = study.value_of(ansatz.default_initial_params())

    # Compute value by simulating time evolution
    if isinstance(hamiltonian, openfermion.DiagonalCoulombHamiltonian):
        quarter_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
                one_body=hamiltonian.one_body,
                two_body=0.25 * hamiltonian.two_body)
        three_quarters_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
                one_body=hamiltonian.one_body,
                two_body=0.75 * hamiltonian.two_body)
    elif isinstance(hamiltonian, openfermion.InteractionOperator):
        quarter_way_hamiltonian = openfermion.InteractionOperator(
                constant=hamiltonian.constant,
                one_body_tensor=hamiltonian.one_body_tensor,
                two_body_tensor=0.25 * hamiltonian.two_body_tensor)
        three_quarters_way_hamiltonian = openfermion.InteractionOperator(
                constant=hamiltonian.constant,
                one_body_tensor=hamiltonian.one_body_tensor,
                two_body_tensor=0.75 * hamiltonian.two_body_tensor)

    simulation_circuit = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                quarter_way_hamiltonian,
                time=0.5 * ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=order,
                algorithm=trotter_algorithm),
            simulate_trotter(
                qubits,
                three_quarters_way_hamiltonian,
                time=0.5 * ansatz.adiabatic_evolution_time,
                n_steps=1,
                order=order,
                algorithm=trotter_algorithm)
    )
    final_state = (preparation_circuit + simulation_circuit
                  ).apply_unitary_effect_to_state()
    correct_val = openfermion.expectation(
            objective._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(obj_val, study_val, atol=atol)
    numpy.testing.assert_allclose(obj_val, correct_val, atol=atol)
