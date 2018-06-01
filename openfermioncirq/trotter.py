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

from typing import Optional, Sequence, Tuple

import numpy

import cirq
from cirq import QubitId, abc
from openfermion import DiagonalCoulombHamiltonian, QuadraticHamiltonian

from openfermioncirq import (CCZ, CXXYY, CYXXY, XXYY, YXXY,
                             bogoliubov_transform, swap_network)


class TrotterStepAlgorithm:
    """An algorithm for a second-order (first-order symmetric) Trotter step.

    This class encapsulates a method for performing a second-order Trotter step.
    It assumes that Hamiltonian evolution using a Trotter-Suzuki product formula
    is performed in the following steps:
        1. Perform some preparatory operations (for instance, a basis change).
        2. Perform a number of Trotter steps. Each Trotter step may induce a
           permutation on the ordering in which qubits represent fermionic
           modes.
        3. Perform some finishing operations.

    Attriubutes:
        controlled: A bool indicating whether the Trotter step is controlled
            by a control qubit.
    """

    controlled = False

    def prepare(self,
                qubits: Sequence[QubitId],
                hamiltonian: DiagonalCoulombHamiltonian,
                control_qubit: Optional[QubitId]=None
                ) -> cirq.OP_TREE:
        """Operations to perform before doing the Trotter steps.

        Args:
            qubits: The qubits on which to perform operations. They should
                be sorted so that the j-th qubit in the Sequence holds the
                occupation of the j-th fermionic mode.
            hamiltonian: The Hamiltonian to simulate.
            control_qubit: The control qubit, if the algorithm is controlled.
        """
        # Default: do nothing
        return ()

    @abc.abstractmethod
    def second_order_trotter_step(
            self,
            qubits: Sequence[QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[QubitId]=None
            ) -> cirq.OP_TREE:
        """Yield operations to perform a second-order Trotter step.

        Args:
            qubits: The qubits on which to apply the Trotter step.
            hamiltonian: The Hamiltonian to simulate.
            time: The evolution time.
            control_qubit: The control qubit, if the algorithm is controlled.
        """
        pass

    def step_qubit_permutation(self,
                               qubits: Sequence[QubitId],
                               hamiltonian: DiagonalCoulombHamiltonian,
                               control_qubit: Optional[QubitId]=None
                               ) -> Tuple[Sequence[QubitId], Optional[QubitId]]:
        """The qubit permutation induced by a single second-order Trotter step.

        Returns:
            A tuple whose first element is the new list of system qubits and
            second element is the new control qubit
        """
        # Default: identity permutation
        return qubits, control_qubit

    def finish(self,
               qubits: Sequence[QubitId],
               hamiltonian: DiagonalCoulombHamiltonian,
               n_steps: int,
               control_qubit: Optional[QubitId]=None
               ) -> cirq.OP_TREE:
        """Operations to perform after all Trotter steps are done."""
        # Default: do nothing
        return ()


class SwapNetworkTrotterStep(TrotterStepAlgorithm):
    """A Trotter step using two consecutive fermionic swap networks.

    This algorithm is described in arXiv:1711.04789.
    """

    def second_order_trotter_step(
            self,
            qubits: Sequence[QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        def one_and_two_body_interaction(p, q, a, b):
            yield XXYY(a, b)**(
                    numpy.real(hamiltonian.one_body[p, q]) * time / numpy.pi)
            yield YXXY(a, b)**(
                    numpy.imag(hamiltonian.one_body[p, q]) * time / numpy.pi)
            yield cirq.CZ(a, b)**(-hamiltonian.two_body[p, q] * time / numpy.pi)

        # Apply one- and two-body interactions for half of the full time
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.Z(qubits[i])**(
                    -numpy.real(hamiltonian.one_body[i, i]) * time / numpy.pi)
               for i in range(n_qubits))

        # Apply one- and two-body interactions for half of the full time
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True,
                           offset=True)


class SplitOperatorTrotterStep(TrotterStepAlgorithm):
    """A Trotter step using a split-operator approach.

    This algorithm is described in arXiv:1706.00023.
    """

    def prepare(self,
                qubits: Sequence[QubitId],
                hamiltonian: DiagonalCoulombHamiltonian,
                control_qubits: Optional[QubitId]=None
                ) -> cirq.OP_TREE:
        #Change to the basis in which the one-body term is diagonal
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)
        yield cirq.inverse_of_invertible_op_tree(
                bogoliubov_transform(
                    qubits, quad_ham.diagonalizing_bogoliubov_transform()))
        # TODO Maybe use FFFT instead

    def second_order_trotter_step(
            self,
            qubits: Sequence[QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)

        # Get the coefficients of the one-body terms in the diagonalizing basis
        orbital_energies, _ = quad_ham.orbital_energies()
        # Get the Bogoliubov transformation matrix that diagonalizes the
        # one-body term
        transformation_matrix = quad_ham.diagonalizing_bogoliubov_transform()

        # Simulate the one-body terms for half of the full time
        yield (cirq.Z(qubits[i])**(-orbital_energies[i] * 0.5 * time / numpy.pi)
               for i in range(n_qubits))

        # Rotate to the computational basis
        yield bogoliubov_transform(qubits, transformation_matrix)

        # Simulate the two-body terms for the full time
        def two_body_interaction(p, q, a, b):
            yield cirq.CZ(a, b)**(
                    -2 * hamiltonian.two_body[p, q] * time / numpy.pi)
        yield swap_network(qubits, two_body_interaction)
        # The qubit ordering has been reversed
        qubits = qubits[::-1]

        # Rotate back to the basis in which the one-body term is diagonal
        yield cirq.inverse_of_invertible_op_tree(
                bogoliubov_transform(qubits, transformation_matrix))

        # Simulate the one-body terms for half of the full time
        yield (cirq.Z(qubits[i])**(-orbital_energies[i] * 0.5 * time / numpy.pi)
               for i in range(n_qubits))

    def step_qubit_permutation(self,
                               qubits: Sequence[QubitId],
                               hamiltonian: DiagonalCoulombHamiltonian,
                               control_qubit: Optional[QubitId]=None
                               ) -> Tuple[Sequence[QubitId], Optional[QubitId]]:
        # A second-order Trotter step reverses the qubit ordering
        return qubits[::-1], None

    def finish(self,
               qubits: Sequence[QubitId],
               hamiltonian: DiagonalCoulombHamiltonian,
               n_steps: int,
               control_qubit: Optional[QubitId]=None
               ) -> cirq.OP_TREE:
        # Rotate back to the computational basis
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)
        yield bogoliubov_transform(
                qubits, quad_ham.diagonalizing_bogoliubov_transform())
        # If the number of Trotter steps is odd, swap qubits back
        if n_steps & 1:
            yield swap_network(qubits)


SWAP_NETWORK = SwapNetworkTrotterStep()
SPLIT_OPERATOR = SplitOperatorTrotterStep()


class ControlledSwapNetworkTrotterStep(TrotterStepAlgorithm):

    controlled = True

    def second_order_trotter_step(
            self,
            qubits: Sequence[QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        def one_and_two_body_interaction(p, q, a, b):
            yield CXXYY(control_qubit, a, b)**(
                    numpy.real(hamiltonian.one_body[p, q]) * time / numpy.pi)
            yield CYXXY(control_qubit, a, b)**(
                    numpy.imag(hamiltonian.one_body[p, q]) * time / numpy.pi)
            yield CCZ(control_qubit, a, b)**(
                    -hamiltonian.two_body[p, q] * time / numpy.pi)

        # Apply one- and two-body interactions for half of the full time
        yield swap_network(
                qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.CZ(control_qubit, qubits[i])**(
                    -numpy.real(hamiltonian.one_body[i, i]) * time / numpy.pi)
               for i in range(n_qubits))

        # Apply one- and two-body interactions for half of the full time
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True,
                           offset=True)


class ControlledSplitOperatorTrotterStep(SplitOperatorTrotterStep):

    controlled = True

    def second_order_trotter_step(
            self,
            qubits: Sequence[QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)

        # Get the coefficients of the one-body terms in the diagonalizing basis
        orbital_energies, _ = quad_ham.orbital_energies()
        # Get the Bogoliubov transformation matrix that diagonalizes the
        # one-body term
        transformation_matrix = quad_ham.diagonalizing_bogoliubov_transform()

        # Simulate the one-body terms for half of the full time
        yield (cirq.CZ(control_qubit, qubits[i])**(
                    -orbital_energies[i] * 0.5 * time / numpy.pi)
               for i in range(n_qubits))

        # Rotate to the computational basis
        yield bogoliubov_transform(qubits, transformation_matrix)

        # Simulate the two-body terms for the full time
        def two_body_interaction(p, q, a, b):
            yield CCZ(control_qubit, a, b)**(
                    -2 * hamiltonian.two_body[p, q] * time / numpy.pi)
        yield swap_network(qubits, two_body_interaction)
        # The qubit ordering has been reversed
        qubits = qubits[::-1]

        # Rotate back to the basis in which the one-body term is diagonal
        yield cirq.inverse_of_invertible_op_tree(
                bogoliubov_transform(qubits, transformation_matrix))

        # Simulate the one-body terms for half of the full time
        yield (cirq.CZ(control_qubit, qubits[i])**(
                    -orbital_energies[i] * 0.5 * time / numpy.pi)
               for i in range(n_qubits))

    def step_qubit_permutation(self,
                               qubits: Sequence[QubitId],
                               hamiltonian: DiagonalCoulombHamiltonian,
                               control_qubit: Optional[QubitId]=None
                               ) -> Tuple[Sequence[QubitId], Optional[QubitId]]:
        # A second-order Trotter step reverses the qubit ordering
        return qubits[::-1], control_qubit


CONTROLLED_SWAP_NETWORK = ControlledSwapNetworkTrotterStep()
CONTROLLED_SPLIT_OPERATOR = ControlledSplitOperatorTrotterStep()


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
