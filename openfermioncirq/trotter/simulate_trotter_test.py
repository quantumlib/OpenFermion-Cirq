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
from openfermion.utils._testing_utils import random_diagonal_coulomb_hamiltonian

from openfermioncirq import simulate_trotter
from openfermioncirq.trotter import (
        SPLIT_OPERATOR,
        SWAP_NETWORK,
        TrotterStepAlgorithm)


def fidelity(state1, state2):
    return abs(numpy.dot(state1, numpy.conjugate(state2)))**2


# Initialize test parameters for a random Hamiltonian
n_qubits = 5
random_hamiltonian = random_diagonal_coulomb_hamiltonian(
        n_qubits, real=False, seed=8440)
random_time = 0.1

# Construct a random initial state
numpy.random.seed(3570)
initial_state = numpy.random.randn(2**n_qubits)
initial_state /= numpy.linalg.norm(initial_state)
initial_state = initial_state.astype(numpy.complex64, copy=False)
assert numpy.allclose(numpy.linalg.norm(initial_state), 1.0)

# Simulate exact evolution
random_sparse = openfermion.get_sparse_operator(random_hamiltonian)
random_exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * random_time * random_sparse, initial_state)

# Make sure the time is not too small
assert fidelity(random_exact_state, initial_state) < .95


@pytest.mark.parametrize(
        'hamiltonian, time, initial_state, exact_state, order, n_steps, '
        'algorithm, controlled, result_fidelity', [
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 0, 3, SWAP_NETWORK, False, .99),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 1, 1, SWAP_NETWORK, False, .999),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 2, 1, SWAP_NETWORK, False, .999999),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 1, 1, SWAP_NETWORK, True, .999),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 1, 1, SPLIT_OPERATOR, False, .9999),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 2, 1, SPLIT_OPERATOR, False, .9999999),
            (random_hamiltonian, random_time, initial_state,
                random_exact_state, 1, 1, SPLIT_OPERATOR, True, .9999),
])
def test_simulate_trotter_simulate(
        hamiltonian, time, initial_state, exact_state, order, n_steps,
        algorithm, controlled, result_fidelity):

    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)
    simulator = cirq.google.XmonSimulator()

    if controlled:
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


def test_simulate_trotter_omit_final_swaps():
    qubits = cirq.LineQubit.range(5)
    hamiltonian = random_diagonal_coulomb_hamiltonian(5, seed=0)
    time = 1.0

    circuit_with_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0, algorithm=SWAP_NETWORK))
    circuit_without_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0, algorithm=SWAP_NETWORK,
                omit_final_swaps=True))

    assert (circuit_with_swaps.to_text_diagram(transpose=True).strip() ==
            (circuit_without_swaps.to_text_diagram(transpose=True).strip() + """
│       ×ᶠ──────────×ᶠ         ×ᶠ───────────×ᶠ
│       │           │          │            │
×ᶠ──────×ᶠ          ×ᶠ─────────×ᶠ           │
│       │           │          │            │
│       ×ᶠ──────────×ᶠ         ×ᶠ───────────×ᶠ
│       │           │          │            │
×ᶠ──────×ᶠ          ×ᶠ─────────×ᶠ           │
│       │           │          │            │
│       ×ᶠ──────────×ᶠ         ×ᶠ───────────×ᶠ
│       │           │          │            │
""").strip())

    circuit_with_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                hamiltonian,
                time,
                order=1,
                n_steps=3,
                algorithm=SPLIT_OPERATOR),
            strategy=cirq.InsertStrategy.NEW)
    circuit_without_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits,
                hamiltonian,
                time,
                order=1,
                n_steps=3,
                algorithm=SPLIT_OPERATOR,
                omit_final_swaps=True),
            strategy=cirq.InsertStrategy.NEW)

    assert (circuit_with_swaps.to_text_diagram(transpose=True).strip() ==
            (circuit_without_swaps.to_text_diagram(transpose=True).strip() + """
│         │            │           ×───────────×
│         │            │           │           │
│         ×────────────×           │           │
│         │            │           │           │
│         │            ×───────────×           │
│         │            │           │           │
×─────────×            │           │           │
│         │            │           │           │
│         │            │           ×───────────×
│         │            │           │           │
│         ×────────────×           │           │
│         │            │           │           │
│         │            ×───────────×           │
│         │            │           │           │
×─────────×            │           │           │
│         │            │           │           │
│         │            │           ×───────────×
│         │            │           │           │
│         ×────────────×           │           │
│         │            │           │           │
""").strip())


def test_simulate_trotter_bad_order_raises_error():
    qubits = cirq.LineQubit.range(2)
    hamiltonian = random_diagonal_coulomb_hamiltonian(2, seed=0)
    time = 1.0
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=-1))


def test_simulate_trotter_bad_hamiltonian_type_raises_error():
    qubits = cirq.LineQubit.range(2)
    hamiltonian = openfermion.FermionOperator()
    time = 1.0
    with pytest.raises(TypeError):
        _ = next(simulate_trotter(qubits, hamiltonian, time,
                                  algorithm=SWAP_NETWORK))


def test_simulate_trotter_unsupported_trotter_step_raises_error():
    qubits = cirq.LineQubit.range(2)
    control = cirq.LineQubit(-1)
    hamiltonian = random_diagonal_coulomb_hamiltonian(2, seed=0)
    time = 1.0
    algorithm = TrotterStepAlgorithm(
            supported_types={openfermion.DiagonalCoulombHamiltonian})
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=0,
                                  algorithm=algorithm))
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=1,
                                  algorithm=algorithm))
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=0,
                                  algorithm=algorithm, control_qubit=control))
    with pytest.raises(ValueError):
        _ = next(simulate_trotter(qubits, hamiltonian, time, order=1,
                                  algorithm=algorithm, control_qubit=control))
