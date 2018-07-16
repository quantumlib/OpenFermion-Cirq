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

from typing import Optional

import os

import numpy
import pytest

import cirq
import openfermion
from openfermion.utils._testing_utils import (
        random_diagonal_coulomb_hamiltonian)

from openfermioncirq import (
        HamiltonianVariationalStudy,
        OptimizationParams,
        SplitOperatorTrotterAnsatz,
        SwapNetworkTrotterAnsatz,
        prepare_gaussian_state,
        simulate_trotter)
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult,
        OptimizationTrialResult)
from openfermioncirq.trotter import LINEAR_SWAP_NETWORK, SPLIT_OPERATOR


class ExampleAlgorithm(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        if initial_guess is None:
            # coverage: ignore
            initial_guess = numpy.ones(black_box.dimension)
        if initial_guess_array is None:
            # coverage: ignore
            initial_guess_array = numpy.ones((3, black_box.dimension))
        a = black_box.evaluate(initial_guess)
        b = black_box.evaluate_with_cost(initial_guess_array[0], 1.0)
        return OptimizationResult(optimal_value=min(a, b),
                                  optimal_parameters=initial_guess,
                                  num_evaluations=1,
                                  cost_spent=0.0,
                                  status=0,
                                  message='success')


# Construct a Hamiltonian for testing
test_hamiltonian = random_diagonal_coulomb_hamiltonian(4, real=True, seed=26191)
test_fermion_op = openfermion.get_fermion_operator(test_hamiltonian)


def test_hamiltonian_variational_study_init_qubit_operator():

    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy(
            'study', ansatz, openfermion.QubitOperator((0, 'X')))
    assert study.hamiltonian == openfermion.QubitOperator((0, 'X'))


def test_hamiltonian_variational_study_noise():

    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, test_hamiltonian)

    numpy.random.seed(10821)
    assert (abs(study.noise()) < abs(study.noise(1e6)) < abs(study.noise(1e5)) <
            abs(study.noise(1e4)) < abs(study.noise(1e3)))


def test_hamiltonian_variational_study_noise_bounds():

    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, test_hamiltonian)

    numpy.random.seed(38017)

    a, b = study.noise_bounds(1e4)
    c, d = study.noise_bounds(1e2)

    numpy.testing.assert_allclose(10 * a, c)
    numpy.testing.assert_allclose(10 * b, d)

    a, b = study.noise_bounds(1e4, confidence=0.95)
    c, d = study.noise_bounds(1e2, confidence=0.95)

    numpy.testing.assert_allclose(10 * a, c)
    numpy.testing.assert_allclose(10 * b, d)

    numpy.testing.assert_allclose(study.noise_bounds(1e2),
                                  study.noise_bounds(1e2, 0.99))

    with pytest.raises(ValueError):
        _ = study.noise_bounds(1.0, 1.0)

    with pytest.raises(ValueError):
        _ = study.noise_bounds(1.0, -1.0)


def test_hamiltonian_variational_study_optimize():
    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy('study',
                                        ansatz,
                                        test_fermion_op)
    study.optimize(
            OptimizationParams(
                ExampleAlgorithm(),
                cost_of_evaluate=1.0),
            'run',
            reevaluate_final_params=True)
    result = study.results['run']
    assert all(result.data_frame['optimal_parameters'].apply(study.evaluate) ==
               result.data_frame['optimal_value'])
    assert result.params.cost_of_evaluate == 1.0


def test_hamiltonian_variational_study_save_load():
    datadir = 'tmp_ffETr2rB49RGP8WE8jer'
    study_name = 'test_hamiltonian_study'


    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy(
            study_name,
            ansatz,
            test_fermion_op,
            datadir=datadir)
    study.optimize(
            OptimizationParams(
                ExampleAlgorithm(),
                cost_of_evaluate=1.0),
            'example')
    study.save()
    loaded_study = HamiltonianVariationalStudy.load(study_name, datadir=datadir)

    assert loaded_study.name == study.name
    assert str(loaded_study.circuit) == str(study.circuit)
    assert loaded_study.datadir == datadir
    assert loaded_study.hamiltonian == test_fermion_op
    assert len(loaded_study.results) == 1

    result = loaded_study.results['example']
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1
    assert result.params.cost_of_evaluate == 1.0

    # Clean up
    os.remove('{}/{}.study'.format(datadir, study_name))
    os.rmdir(datadir)


def test_swap_network_trotter_ansatz_value_not_implemented():
    ansatz = SwapNetworkTrotterAnsatz(test_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, test_hamiltonian)
    trial_result = cirq.TrialResult(
            params=ansatz.param_resolver(ansatz.default_initial_params()),
            measurements={},
            repetitions=1)
    with pytest.raises(NotImplementedError):
        _ = study.value(trial_result)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, test_hamiltonian, 5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, test_hamiltonian, 5e-5),
])
def test_trotter_ansatzes_evaluate_order_1(
        ansatz_factory, trotter_algorithm, hamiltonian, atol):
    """Check that a Trotter aansatz with one iteration and default parameters
    is consistent with time evolution with one Trotter step."""

    ansatz = ansatz_factory(hamiltonian, iterations=1)
    qubits = ansatz.qubits

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )
    study = HamiltonianVariationalStudy('study',
                                        ansatz,
                                        hamiltonian,
                                        preparation_circuit=preparation_circuit)

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz
    val = study.evaluate(study.default_initial_params())

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
    circuit = preparation_circuit + simulation_circuit
    result = simulator.simulate(circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            study._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(val, correct_val, atol=atol)


@pytest.mark.parametrize(
        'ansatz_factory, trotter_algorithm, hamiltonian, atol', [
    (SwapNetworkTrotterAnsatz, LINEAR_SWAP_NETWORK, test_hamiltonian, 5e-5),
    (SplitOperatorTrotterAnsatz, SPLIT_OPERATOR, test_hamiltonian, 5e-5),
])
def test_trotter_ansatzes_evaluate_order_2(
        ansatz_factory, trotter_algorithm, hamiltonian, atol):
    """Check that a Trotter ansatz with two iterations and default parameters
    is consistent with time evolution with two Trotter steps."""

    ansatz = ansatz_factory(hamiltonian, iterations=2)
    qubits = ansatz.qubits

    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(qubits) // 2))
    )
    study = HamiltonianVariationalStudy('study',
                                        ansatz,
                                        hamiltonian,
                                        preparation_circuit=preparation_circuit)

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz
    val = study.evaluate(study.default_initial_params())

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
    circuit = preparation_circuit + simulation_circuit
    result = simulator.simulate(circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            study._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(val, correct_val, atol=atol)
