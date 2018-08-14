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

from typing import Callable, Optional, Tuple

import numpy
import pytest
import scipy.sparse.linalg

import cirq
import openfermion
from openfermion.utils._testing_utils import (
        random_diagonal_coulomb_hamiltonian,
        random_interaction_operator)

from openfermioncirq import simulate_trotter
from openfermioncirq.trotter import (
        SPLIT_OPERATOR,
        LINEAR_SWAP_NETWORK,
        LOW_RANK,
        TrotterAlgorithm)
from openfermioncirq.trotter.trotter_algorithm import Hamiltonian


def fidelity(state1, state2):
    return abs(numpy.dot(state1, numpy.conjugate(state2)))**2


def produce_simulation_test_parameters(
        n_qubits: int,
        time: float,
        hamiltonian_factory: Callable[[int, Optional[bool]], Hamiltonian],
        real: bool,
        seed: Optional[int]=None
        ) -> Tuple[Hamiltonian, numpy.ndarray, numpy.ndarray]:
    """Produce objects for testing Hamiltonian simulation.

    Constructs a Hamiltonian with the given parameters, produces a random
    initial state, and evolves the initial state for the specified amount of
    time. Returns the constructed Hamiltonian, the initial state, and the
    final state.

    Args:
        n_qubits: The number of qubits of the Hamiltonian
        time: The time to evolve for
        hamiltonian_factory: A Callable that takes a takes two arguments,
            (n_qubits, real) giving the number of qubits and whether
            to use only real numbers, and returns a Hamiltonian.
        real: Whether the Hamiltonian should use only real numbers
        seed: an RNG seed.
    """

    numpy.random.seed(seed)

    # Construct a random initial state
    initial_state = numpy.random.randn(2**n_qubits)
    initial_state /= numpy.linalg.norm(initial_state)
    initial_state = initial_state.astype(numpy.complex64, copy=False)
    assert numpy.allclose(numpy.linalg.norm(initial_state), 1.0)

    # Construct a Hamiltonian
    hamiltonian = hamiltonian_factory(n_qubits, real=real)  # type: ignore

    # Simulate exact evolution
    hamiltonian_sparse = openfermion.get_sparse_operator(hamiltonian)
    exact_state = scipy.sparse.linalg.expm_multiply(
            -1j * time * hamiltonian_sparse, initial_state)

    # Make sure the time is not too small
    assert fidelity(exact_state, initial_state) < .95

    return hamiltonian, initial_state, exact_state


big_time = 0.1
small_time = 0.05

diag_coul_hamiltonian, diag_coul_initial_state, diag_coul_exact_state = (
        produce_simulation_test_parameters(
            4, big_time, random_diagonal_coulomb_hamiltonian,
            real=False, seed=49075))

interaction_op3, interaction_op_initial_state3, interaction_op_exact_state3 = (
        produce_simulation_test_parameters(
            3, big_time, random_interaction_operator,
            real=True, seed=48565))

interaction_op4, interaction_op_initial_state4, interaction_op_exact_state4 = (
        produce_simulation_test_parameters(
            4, small_time, random_interaction_operator,
            real=True, seed=19372))


@pytest.mark.parametrize(
        'hamiltonian, time, initial_state, exact_state, order, n_steps, '
        'algorithm, result_fidelity', [
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 1, None, .98),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 2, None, .99),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 1, 1, LINEAR_SWAP_NETWORK, .999),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 2, 1, LINEAR_SWAP_NETWORK, .999999),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 1, SPLIT_OPERATOR, .996),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 2, SPLIT_OPERATOR, .999),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 1, 1, SPLIT_OPERATOR, .9999),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 2, 1, SPLIT_OPERATOR, .9999999),
            (interaction_op3, big_time, interaction_op_initial_state3,
                interaction_op_exact_state3, 0, 1, None, .998),
            (interaction_op3, big_time, interaction_op_initial_state3,
                interaction_op_exact_state3, 0, 2, LOW_RANK, .999),
            (interaction_op4, small_time, interaction_op_initial_state4,
                interaction_op_exact_state4, 0, 1, LOW_RANK, .991),
            (interaction_op4, small_time, interaction_op_initial_state4,
                interaction_op_exact_state4, 0, 2, LOW_RANK, .998),
])
def test_simulate_trotter_simulate(
        hamiltonian, time, initial_state, exact_state, order, n_steps,
        algorithm, result_fidelity):

    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    start_state = initial_state

    circuit = cirq.Circuit.from_ops(simulate_trotter(
        qubits, hamiltonian, time, n_steps, order, algorithm))

    final_state = circuit.apply_unitary_effect_to_state(start_state)
    correct_state = exact_state
    assert fidelity(final_state, correct_state) > result_fidelity
    # Make sure the time wasn't too small
    assert fidelity(final_state, start_state) < 0.95 * result_fidelity


@pytest.mark.parametrize(
        'hamiltonian, time, initial_state, exact_state, order, n_steps, '
        'algorithm, result_fidelity', [
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 1, LINEAR_SWAP_NETWORK, .993),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 1, 1, LINEAR_SWAP_NETWORK, .999),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 0, 1, SPLIT_OPERATOR, .998),
            (diag_coul_hamiltonian, big_time, diag_coul_initial_state,
                diag_coul_exact_state, 1, 1, SPLIT_OPERATOR, .9999),
            (interaction_op3, big_time, interaction_op_initial_state3,
                interaction_op_exact_state3, 0, 1, LOW_RANK, .9991),
            (interaction_op3, big_time, interaction_op_initial_state3,
                interaction_op_exact_state3, 0, 2, LOW_RANK, .9998),
            (interaction_op4, small_time, interaction_op_initial_state4,
                interaction_op_exact_state4, 0, 1, LOW_RANK, .995),
            (interaction_op4, small_time, interaction_op_initial_state4,
                interaction_op_exact_state4, 0, 2, LOW_RANK, .998),
])
def test_simulate_trotter_simulate_controlled(
        hamiltonian, time, initial_state, exact_state, order, n_steps,
        algorithm, result_fidelity):

    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    control = cirq.LineQubit(-1)
    zero = [1, 0]
    one = [0, 1]
    start_state = (numpy.kron(zero, initial_state)
                   + numpy.kron(one, initial_state)) / numpy.sqrt(2)

    circuit = cirq.Circuit.from_ops(simulate_trotter(
        qubits, hamiltonian, time, n_steps, order, algorithm, control))

    final_state = circuit.apply_unitary_effect_to_state(start_state)
    correct_state = (numpy.kron(zero, initial_state)
                     + numpy.kron(one, exact_state)) / numpy.sqrt(2)
    assert fidelity(final_state, correct_state) > result_fidelity
    # Make sure the time wasn't too small
    assert fidelity(final_state, start_state) < 0.95 * result_fidelity


def test_simulate_trotter_omit_final_swaps():
    n_qubits = 5
    qubits = cirq.LineQubit.range(n_qubits)
    hamiltonian = openfermion.DiagonalCoulombHamiltonian(
            one_body=numpy.ones((n_qubits, n_qubits)),
            two_body=numpy.ones((n_qubits, n_qubits)))
    time = 1.0

    circuit_with_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0,
                algorithm=LINEAR_SWAP_NETWORK))
    circuit_without_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0,
                algorithm=LINEAR_SWAP_NETWORK,
                omit_final_swaps=True))

    assert (circuit_with_swaps.to_text_diagram(transpose=True).strip() ==
            (circuit_without_swaps.to_text_diagram(transpose=True).strip() + """
│        ×ᶠ─────────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ         │
│        │          │          │          │
│        ×ᶠ─────────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │          │
×ᶠ───────×ᶠ         ×ᶠ─────────×ᶠ         │
│        │          │          │          │
│        ×ᶠ─────────×ᶠ         ×ᶠ─────────×ᶠ
│        │          │          │          │
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
│         │           │           ×────────────×
│         │           │           │            │
│         ×───────────×           │            │
│         │           │           │            │
│         │           ×───────────×            │
│         │           │           │            │
×─────────×           │           │            │
│         │           │           │            │
│         │           │           ×────────────×
│         │           │           │            │
│         ×───────────×           │            │
│         │           │           │            │
│         │           ×───────────×            │
│         │           │           │            │
×─────────×           │           │            │
│         │           │           │            │
│         │           │           ×────────────×
│         │           │           │            │
│         ×───────────×           │            │
│         │           │           │            │
""").strip())

    hamiltonian = random_interaction_operator(n_qubits, seed=0)
    circuit_with_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0,
                algorithm=LOW_RANK))
    circuit_without_swaps = cirq.Circuit.from_ops(
            simulate_trotter(
                qubits, hamiltonian, time, order=0,
                algorithm=LOW_RANK,
                omit_final_swaps=True))

    assert len(circuit_without_swaps) < len(circuit_with_swaps)


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
                                  algorithm=None))
    with pytest.raises(TypeError):
        _ = next(simulate_trotter(qubits, hamiltonian, time,
                                  algorithm=LINEAR_SWAP_NETWORK))


def test_simulate_trotter_unsupported_trotter_step_raises_error():
    qubits = cirq.LineQubit.range(2)
    control = cirq.LineQubit(-1)
    hamiltonian = random_diagonal_coulomb_hamiltonian(2, seed=0)
    time = 1.0
    class EmptyTrotterAlgorithm(TrotterAlgorithm):
        supported_types = {openfermion.DiagonalCoulombHamiltonian}
    algorithm = EmptyTrotterAlgorithm()
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
