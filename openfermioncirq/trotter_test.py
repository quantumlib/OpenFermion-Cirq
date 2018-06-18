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
from cirq import LineQubit
from openfermion import (
        QuadraticHamiltonian,
        count_qubits,
        fermi_hubbard,
        get_diagonal_coulomb_hamiltonian,
        get_sparse_operator,
        jw_get_gaussian_state)

from openfermioncirq.trotter import (
        CONTROLLED_SPLIT_OPERATOR,
        CONTROLLED_SWAP_NETWORK,
        SPLIT_OPERATOR,
        SWAP_NETWORK,
        simulate_trotter)


def fidelity(state1, state2):
    return abs(numpy.dot(state1, numpy.conjugate(state2)))


# Initialize test parameters for a Hubbard Hamiltonian
hubbard_hamiltonian = get_diagonal_coulomb_hamiltonian(
        fermi_hubbard(2, 2, 1., 4.))
# Use mean-field initial state and energy scale
quad_ham = QuadraticHamiltonian(hubbard_hamiltonian.one_body)
hubbard_energy, hubbard_initial_state = jw_get_gaussian_state(quad_ham)
hubbard_initial_state = hubbard_initial_state.astype(
        numpy.complex64, copy=False)
assert numpy.allclose(numpy.linalg.norm(hubbard_initial_state), 1.0)
hubbard_time = abs(hubbard_energy) / 10
# Simulate exact evolution
hubbard_sparse = get_sparse_operator(hubbard_hamiltonian)
hubbard_exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * hubbard_time * hubbard_sparse, hubbard_initial_state)
# Make sure the time is not too small
assert fidelity(hubbard_exact_state, hubbard_initial_state) < .91


# Initialize test parameters for a Hamiltonian with complex entries
complex_hamiltonian = get_diagonal_coulomb_hamiltonian(
        fermi_hubbard(2, 2, 1., 4.))
complex_hamiltonian.one_body += 1j * numpy.triu(complex_hamiltonian.one_body)
complex_hamiltonian.one_body -= 1j * numpy.tril(complex_hamiltonian.one_body)
# Use mean-field initial state and energy scale
quad_ham = QuadraticHamiltonian(complex_hamiltonian.one_body)
complex_energy, complex_initial_state = jw_get_gaussian_state(quad_ham)
complex_initial_state = complex_initial_state.astype(
        numpy.complex64, copy=False)
assert numpy.allclose(numpy.linalg.norm(complex_initial_state), 1.0)
complex_time = abs(complex_energy) / 15
# Simulate exact evolution
complex_sparse = get_sparse_operator(complex_hamiltonian)
complex_exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * complex_time * complex_sparse, complex_initial_state)
# Make sure the time is not too small
assert fidelity(complex_exact_state, complex_initial_state) < .94


@pytest.mark.parametrize(
        'hamiltonian, time, initial_state, exact_state, order, n_steps, '
        'algorithm, result_fidelity', [
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  1, 3, SWAP_NETWORK, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  2, 1, SWAP_NETWORK, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  1, 3, SPLIT_OPERATOR, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  2, 1, SPLIT_OPERATOR, .99),
            (complex_hamiltonian, complex_time, complex_initial_state,
                complex_exact_state,  1, 3, SWAP_NETWORK, .99),
            (complex_hamiltonian, complex_time, complex_initial_state,
                complex_exact_state,  1, 3, SPLIT_OPERATOR, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  1, 3, CONTROLLED_SWAP_NETWORK, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  2, 1, CONTROLLED_SWAP_NETWORK, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  1, 3, CONTROLLED_SPLIT_OPERATOR, .99),
            (hubbard_hamiltonian, hubbard_time, hubbard_initial_state,
                hubbard_exact_state,  2, 1, CONTROLLED_SPLIT_OPERATOR, .99),
            (complex_hamiltonian, complex_time, complex_initial_state,
                complex_exact_state,  1, 3, CONTROLLED_SWAP_NETWORK, .99),
            (complex_hamiltonian, complex_time, complex_initial_state,
                complex_exact_state,  1, 3, CONTROLLED_SPLIT_OPERATOR, .99),
])
def test_simulate_trotter(
        hamiltonian, time, initial_state, exact_state, order, n_steps,
        algorithm, result_fidelity):

    n_qubits = count_qubits(hamiltonian)
    qubits = LineQubit.range(n_qubits)
    simulator = cirq.google.XmonSimulator()

    if algorithm.controlled:
        control = LineQubit(-1)
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
