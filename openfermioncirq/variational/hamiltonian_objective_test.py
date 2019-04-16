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
from openfermion import random_diagonal_coulomb_hamiltonian

from openfermioncirq import HamiltonianObjective


# Construct a Hamiltonian for testing
test_hamiltonian = random_diagonal_coulomb_hamiltonian(4, real=True, seed=26191)
test_fermion_op = openfermion.get_fermion_operator(test_hamiltonian)


def test_hamiltonian_objective_value():

    obj = HamiltonianObjective(test_hamiltonian)
    obj_linear_op = HamiltonianObjective(test_hamiltonian, use_linear_op=True)
    hamiltonian_sparse = openfermion.get_sparse_operator(test_hamiltonian)

    simulator = cirq.Simulator()
    qubits = cirq.LineQubit.range(4)
    numpy.random.seed(10581)
    result = simulator.simulate(
            cirq.testing.random_circuit(qubits, 5, 0.8),
            qubit_order=qubits)
    correct_val = openfermion.expectation(hamiltonian_sparse,
                                          result.final_state)

    numpy.testing.assert_allclose(
            obj.value(result), correct_val, atol=1e-5)
    numpy.testing.assert_allclose(
            obj.value(result.final_state), correct_val, 1e-5)
    numpy.testing.assert_allclose(
            obj_linear_op.value(result), correct_val, 1e-5)
    numpy.testing.assert_allclose(
            obj_linear_op.value(result.final_state), correct_val, 1e-5)


def test_hamiltonian_objective_noise():

    obj = HamiltonianObjective(test_hamiltonian)

    numpy.random.seed(10821)
    assert (abs(obj.noise()) < abs(obj.noise(1e6)) < abs(obj.noise(1e5)) <
            abs(obj.noise(1e4)) < abs(obj.noise(1e3)))


def test_hamiltonian_objective_noise_bounds():

    obj = HamiltonianObjective(test_hamiltonian)

    numpy.random.seed(38017)

    a, b = obj.noise_bounds(1e4)
    c, d = obj.noise_bounds(1e2)

    numpy.testing.assert_allclose(10 * a, c)
    numpy.testing.assert_allclose(10 * b, d)

    a, b = obj.noise_bounds(1e4, confidence=0.95)
    c, d = obj.noise_bounds(1e2, confidence=0.95)

    numpy.testing.assert_allclose(10 * a, c)
    numpy.testing.assert_allclose(10 * b, d)

    numpy.testing.assert_allclose(obj.noise_bounds(1e2),
                                  obj.noise_bounds(1e2, 0.99))

    with pytest.raises(ValueError):
        _ = obj.noise_bounds(1.0, 1.0)

    with pytest.raises(ValueError):
        _ = obj.noise_bounds(1.0, -1.0)


def test_hamiltonian_objective_value_not_implemented():
    obj = HamiltonianObjective(test_hamiltonian)
    trial_result = cirq.TrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            repetitions=1)
    with pytest.raises(NotImplementedError):
        _ = obj.value(trial_result)


def test_hamiltonian_objective_init_qubit_operator():

    obj = HamiltonianObjective(openfermion.QubitOperator((0, 'X')))
    assert obj.hamiltonian == openfermion.QubitOperator((0, 'X'))
