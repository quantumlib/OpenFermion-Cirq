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

from openfermioncirq import prepare_gaussian_state, simulate_trotter
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)
from openfermioncirq.trotter import SWAP_NETWORK

from openfermioncirq.variational.swap_network_trotter_ansatz import (
        SwapNetworkTrotterAnsatz)
from openfermioncirq.variational.hamiltonian_variational_study import (
        HamiltonianVariationalStudy)


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


# Construct a Hubbard model Hamiltonian
hubbard_model = openfermion.fermi_hubbard(2, 2, 1., 4.)
hubbard_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(
        hubbard_model)

# Construct a Hamiltonian with complex one-body entries
grid = openfermion.Grid(2, 2, 1.0)
jellium = openfermion.jellium_model(grid, spinless=True, plane_wave=False)
complex_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)
complex_hamiltonian.one_body += 1j * numpy.triu(complex_hamiltonian.one_body)
complex_hamiltonian.one_body -= 1j * numpy.tril(complex_hamiltonian.one_body)

# Construct an empty Hamiltonian
zero_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
        one_body=numpy.zeros((5, 5)),
        two_body=numpy.zeros((5, 5)))


def test_hamiltonian_variational_study_init_qubit_operator():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    study = HamiltonianVariationalStudy(
            'study', ansatz, openfermion.QubitOperator('X0'))
    assert study.hamiltonian == openfermion.QubitOperator('X0')


def test_hamiltonian_variational_study_noise():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, hubbard_hamiltonian)

    numpy.random.seed(10821)
    assert (abs(study.noise()) < abs(study.noise(1e6)) < abs(study.noise(1e5)) <
            abs(study.noise(1e4)) < abs(study.noise(1e3)))


def test_hamiltonian_variational_study_noise_bound():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, hubbard_hamiltonian)

    numpy.testing.assert_allclose(10 * study.noise_bound(1e4),
                                  study.noise_bound(1e2))
    numpy.testing.assert_allclose(study.noise_bound(), 0)

    with pytest.raises(ValueError):
        _ = study.noise_bound(1.0, 1.0)

    with pytest.raises(ValueError):
        _ = study.noise_bound(1.0, -1.0)


def test_hamiltonian_variational_study_save_load():
    datadir = 'tmp_ffETr2rB49RGP8WE8jer'
    study_name = 'test_hamiltonian_study'

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    study = HamiltonianVariationalStudy(study_name,
                                        ansatz,
                                        hubbard_model,
                                        datadir=datadir)
    study.run('example', ExampleAlgorithm())
    study.save()
    loaded_study = HamiltonianVariationalStudy.load(study_name, datadir=datadir)

    assert loaded_study.name == study.name
    assert str(loaded_study.circuit) == str(study.circuit)
    assert len(loaded_study.results) == 1
    assert isinstance(loaded_study.results['example'][0], OptimizationResult)
    assert loaded_study.datadir == datadir
    assert loaded_study.hamiltonian == hubbard_model

    # Clean up
    os.remove('{}/{}.study'.format(datadir, study_name))
    os.rmdir(datadir)


def test_swap_network_trotter_ansatz_value_not_implemented():
    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, hubbard_hamiltonian)
    trial_result = cirq.TrialResult(
            params=ansatz.param_resolver(ansatz.default_initial_params()),
            measurements={},
            repetitions=1)
    with pytest.raises(NotImplementedError):
        _ = study.value(trial_result)


@pytest.mark.parametrize('hamiltonian, include_all_xxyy, atol',
                         [(hubbard_hamiltonian, True, 5e-6),
                          (complex_hamiltonian, False, 5e-5)])
def test_swap_network_trotter_ansatz_evaluate_order_1(hamiltonian,
                                                      include_all_xxyy,
                                                      atol):
    """Check that the ansatz with one iteration and default parameters is
    consistent with time evolution with one Trotter step."""

    ansatz = SwapNetworkTrotterAnsatz(hamiltonian,
                                      iterations=1,
                                      include_all_xxyy=include_all_xxyy)
    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                ansatz.qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(ansatz.qubits) // 2))
    )
    study = HamiltonianVariationalStudy('study',
                                        ansatz,
                                        hamiltonian,
                                        preparation_circuit=preparation_circuit)

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz
    result = simulator.simulate(
            study.circuit,
            param_resolver=ansatz.param_resolver(
                study.default_initial_params())
    )
    val = study.value(result)

    # Compute value by simulating time evolution
    qubits = cirq.LineQubit.range(len(ansatz.qubits))
    half_way_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
            one_body=hamiltonian.one_body,
            two_body=0.5 * hamiltonian.two_body)
    simulation_circuit = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                half_way_hamiltonian,
                time=100.,
                n_steps=1,
                order=1,
                algorithm=SWAP_NETWORK)
    )
    circuit = preparation_circuit + simulation_circuit
    result = simulator.simulate(circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            study._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(val, correct_val, atol=atol)


@pytest.mark.parametrize('hamiltonian, include_all_xxyy, atol',
                         [(hubbard_hamiltonian, False, 1e-6),
                          (complex_hamiltonian, True, 5e-5)])
def test_swap_network_trotter_ansatz_evaluate_order_2(hamiltonian,
                                                      include_all_xxyy,
                                                      atol):
    """Check that the ansatz with two iterations and default parameters is
    consistent with time evolution with two Trotter steps."""

    ansatz = SwapNetworkTrotterAnsatz(hamiltonian,
                                      iterations=2,
                                      include_all_xxyy=include_all_xxyy)
    preparation_circuit = cirq.Circuit.from_ops(
            prepare_gaussian_state(
                ansatz.qubits,
                openfermion.QuadraticHamiltonian(hamiltonian.one_body),
                occupied_orbitals=range(len(ansatz.qubits) // 2))
    )
    study = HamiltonianVariationalStudy('study',
                                        ansatz,
                                        hamiltonian,
                                        preparation_circuit=preparation_circuit)

    simulator = cirq.google.XmonSimulator()

    # Compute value using ansatz
    result = simulator.simulate(
            study.circuit,
            param_resolver=ansatz.param_resolver(
                study.default_initial_params())
    )
    val = study.value(result)

    # Compute value by simulating time evolution
    qubits = cirq.LineQubit.range(len(ansatz.qubits))
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
                time=50.,
                n_steps=1,
                order=1,
                algorithm=SWAP_NETWORK),
            simulate_trotter(
                qubits,
                three_quarters_way_hamiltonian,
                time=50.,
                n_steps=1,
                order=1,
                algorithm=SWAP_NETWORK)
    )
    circuit = preparation_circuit + simulation_circuit
    result = simulator.simulate(circuit)
    final_state = result.final_state
    correct_val = openfermion.expectation(
            study._hamiltonian_linear_op, final_state).real

    numpy.testing.assert_allclose(val, correct_val, atol=atol)
