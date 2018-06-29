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

from typing import Optional, Sequence

import cirq
from cirq import QubitId
from openfermion import DiagonalCoulombHamiltonian

from openfermioncirq.trotter.trotter_step_algorithm import TrotterStepAlgorithm
from openfermioncirq.trotter.swap_network_trotter_step import SWAP_NETWORK


def simulate_trotter(qubits: Sequence[QubitId],
                     hamiltonian: DiagonalCoulombHamiltonian,
                     time: float,
                     n_steps: int=1,
                     order: int=1,
                     algorithm: TrotterStepAlgorithm=SWAP_NETWORK,
                     control_qubit: Optional[QubitId]=None
                     ) -> cirq.OP_TREE:
    r"""Simulate Hamiltonian evolution using a Trotter-Suzuki product formula.

    The input is a Hamiltonian of the form

    .. math::

        \sum_{p, q} T_{pq} a^\dagger_p a_q + \sum_{p, q} V_{pq} n_p n_q

    where :math:`n_p` denotes the occupation number operator,
    :math:`n_p = a^\dagger_p a_p`.
    The product formula used is from "General theory of fractal path integrals
    with applications to many-body theories and statistical physics" by
    Masuo Suzuki.

    Args:
        qubits: The qubits on which to apply operations. They should be sorted
            so that the j-th qubit in the Sequence holds the occupation of the
            j-th fermionic mode. Any control qubits used by the chosen algorithm
            should be placed at the end.
        hamiltonian: The Hamiltonian to simulate.
        time: The evolution time.
        n_steps: The number of Trotter steps to use.
        order: The order of the product formula. The value indexes symmetric
            formulae, i.e., a value of 1 indicates the first-order symmetric,
            commonly known as the second-order, Trotter step.
        algorithm: A string indicating the algorithm to use to simulate a single
            Trotter step.
            Available options:
                SWAP_NETWORK: The algorithm from arXiv:1711.04789.
                SPLIT_OPERATOR: The algorithm from arXiv:1706.00023.
                CONTROLLED_SWAP_NETWORK: Controlled version of SWAP_NETWORK
                CONTROLLED_SPLIT_OPERATOR: Controlled version of SPLIT_OPERATOR
    """
    # TODO Document gate complexities of algorithm options
    if order < 0:
        raise ValueError('The order of the Trotter formula must be at least 0.')

    # Get ready to perform Trotter steps
    yield algorithm.prepare(qubits, hamiltonian, control_qubit)

    # Perform Trotter steps
    step_time = time / n_steps
    for _ in range(n_steps):
        yield _trotter_step(qubits, hamiltonian, step_time, order, algorithm,
                            control_qubit)
        qubits, control_qubit = algorithm.step_qubit_permutation(
                qubits, hamiltonian, control_qubit)

    # Finish
    yield algorithm.finish(qubits, hamiltonian, n_steps, control_qubit)


def _trotter_step(qubits: Sequence[QubitId],
                  hamiltonian: DiagonalCoulombHamiltonian,
                  time: float,
                  order: int,
                  algorithm: TrotterStepAlgorithm,
                  control_qubit: Optional[QubitId]) -> cirq.OP_TREE:
    """Apply a Trotter step."""
    if order == 0:
        # TODO Yield first-order (asymmetric) formula
        pass

    elif order == 1:
        yield algorithm.second_order_trotter_step(qubits, hamiltonian, time,
                                                  control_qubit)

    else:
        split_time = time / (4 - 4**(1 / (2 * order - 1)))

        for _ in range(2):
            yield _trotter_step(qubits, hamiltonian, split_time, order - 1,
                                algorithm, control_qubit)
            qubits, control_qubit = algorithm.step_qubit_permutation(
                    qubits, hamiltonian, control_qubit)

        yield _trotter_step(
                qubits, hamiltonian, time - 4 * split_time, order - 1,
                algorithm, control_qubit)
        qubits, control_qubit = algorithm.step_qubit_permutation(
                qubits, hamiltonian, control_qubit)

        for _ in range(2):
            yield _trotter_step(qubits, hamiltonian, split_time, order - 1,
                                algorithm, control_qubit)
            qubits, control_qubit = algorithm.step_qubit_permutation(
                    qubits, hamiltonian, control_qubit)
