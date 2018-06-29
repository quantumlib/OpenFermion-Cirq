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
import scipy.sparse.linalg

import cirq
import openfermion

from openfermioncirq.trotter import (
        CONTROLLED_SPLIT_OPERATOR,
        CONTROLLED_SWAP_NETWORK,
        SPLIT_OPERATOR,
        SWAP_NETWORK,
        simulate_trotter)


def fidelity(state1, state2):
    return abs(numpy.dot(state1, numpy.conjugate(state2)))


# Construct a jellium model
dim = 2
length = 2
n_qubits = length**dim
grid = openfermion.Grid(dim, length, 1.0)
jellium = openfermion.jellium_model(grid, spinless=True, plane_wave=False) 


# Construct a random initial state
numpy.random.seed(3570)
initial_state = numpy.random.randn(2**n_qubits)
initial_state /= numpy.linalg.norm(initial_state)
initial_state = initial_state.astype(numpy.complex64, copy=False)
assert numpy.allclose(numpy.linalg.norm(initial_state), 1.0)


# Initialize test parameters for a jellium Hamiltonian
jellium_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)
jellium_time = 0.1
# Simulate exact evolution
jellium_sparse = openfermion.get_sparse_operator(jellium_hamiltonian)
jellium_exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * jellium_time * jellium_sparse, initial_state)
# Make sure the time is not too small
assert fidelity(jellium_exact_state, initial_state) < .95


# Initialize test parameters for a Hamiltonian with complex entries
complex_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)
complex_hamiltonian.one_body += 1j * numpy.triu(complex_hamiltonian.one_body)
complex_hamiltonian.one_body -= 1j * numpy.tril(complex_hamiltonian.one_body)
complex_time = 0.05
# Simulate exact evolution
complex_sparse = openfermion.get_sparse_operator(complex_hamiltonian)
complex_exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * complex_time * complex_sparse, initial_state)
# Make sure the time is not too small
assert fidelity(complex_exact_state, initial_state) < .95


@pytest.mark.parametrize(
        'hamiltonian, time, initial_state, exact_state, order, n_steps, '
        'algorithm, result_fidelity', [
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  1, 3, SWAP_NETWORK, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  2, 1, SWAP_NETWORK, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  1, 3, SPLIT_OPERATOR, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  2, 1, SPLIT_OPERATOR, .99),
            (complex_hamiltonian, complex_time, initial_state,
                complex_exact_state,  1, 3, SWAP_NETWORK, .99),
            (complex_hamiltonian, complex_time, initial_state,
                complex_exact_state,  1, 3, SPLIT_OPERATOR, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  1, 3, CONTROLLED_SWAP_NETWORK, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  2, 1, CONTROLLED_SWAP_NETWORK, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  1, 3, CONTROLLED_SPLIT_OPERATOR, .99),
            (jellium_hamiltonian, jellium_time, initial_state,
                jellium_exact_state,  2, 1, CONTROLLED_SPLIT_OPERATOR, .99),
            (complex_hamiltonian, complex_time, initial_state,
                complex_exact_state,  1, 3, CONTROLLED_SWAP_NETWORK, .99),
            (complex_hamiltonian, complex_time, initial_state,
                complex_exact_state,  1, 3, CONTROLLED_SPLIT_OPERATOR, .99),
])
def test_simulate_trotter(
        hamiltonian, time, initial_state, exact_state, order, n_steps,
        algorithm, result_fidelity):

    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)
    simulator = cirq.google.XmonSimulator()

    if algorithm.controlled:
        control = cirq.LineQubit(-1)
        circuit = cirq.Circuit.from_ops(simulate_trotter(
            qubits, hamiltonian, time, n_steps, order, algorithm, control))

        # With control on
        one = [0, 1]
        start_state = numpy.kron(one, initial_state).astype(
                numpy.complex64, copy=False)
        result = simulator.simulate(circuit,
                                    qubit_order=[control] + qubits,
                                    initial_state=start_state)
        final_state = result.final_state
        correct_state = numpy.kron(one, exact_state)
        assert fidelity(final_state, correct_state) > result_fidelity
        # Make sure the time wasn't too small
        assert fidelity(final_state, start_state) < result_fidelity

        # With control off
        zero = [1, 0]
        start_state = numpy.kron(zero, initial_state).astype(
                numpy.complex64, copy=False)
        result = simulator.simulate(circuit,
                                    qubit_order=[control] + qubits,
                                    initial_state=start_state)
        final_state = result.final_state
        correct_state = start_state
        assert fidelity(final_state, correct_state) > result_fidelity
    else:
        circuit = cirq.Circuit.from_ops(simulate_trotter(
            qubits, hamiltonian, time, n_steps, order, algorithm))
        start_state = initial_state.astype(numpy.complex64, copy=False)
        result = simulator.simulate(circuit,
                                    qubit_order=qubits,
                                    initial_state=start_state)
        final_state = result.final_state
        correct_state = exact_state
        assert fidelity(final_state, correct_state) > result_fidelity
        # Make sure the time wasn't too small
        assert fidelity(final_state, start_state) < result_fidelity


def test_simulate_trotter_bad_order_raises_error():
    qubits = cirq.LineQubit.range(2)
    hamiltonian = jellium_hamiltonian
    time = 1.0
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=-1))
