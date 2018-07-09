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
from openfermion import DiagonalCoulombHamiltonian, QuadraticHamiltonian

from openfermioncirq import CCZ, bogoliubov_transform, swap_network

from openfermioncirq.trotter.trotter_step_algorithm import (
        TrotterStep,
        TrotterStepAlgorithm)


class SymmetricSplitOperatorTrotterStep(TrotterStep):
    """A Trotter step using a split-operator approach.

    This algorithm is described in arXiv:1706.00023.
    """

    def prepare(self,
                qubits: Sequence[cirq.QubitId],
                hamiltonian: DiagonalCoulombHamiltonian,
                control_qubits: Optional[cirq.QubitId]=None
                ) -> cirq.OP_TREE:
        #Change to the basis in which the one-body term is diagonal
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)
        yield cirq.inverse_of_invertible_op_tree(
                bogoliubov_transform(
                    qubits, quad_ham.diagonalizing_bogoliubov_transform()))
        # TODO Maybe use FFFT instead

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
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
        def two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
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
                               qubits: Sequence[cirq.QubitId],
                               control_qubit: Optional[cirq.QubitId]=None
                               ) -> Tuple[Sequence[cirq.QubitId],
                                          Optional[cirq.QubitId]]:
        # A Trotter step reverses the qubit ordering
        return qubits[::-1], None

    def finish(self,
               qubits: Sequence[cirq.QubitId],
               hamiltonian: DiagonalCoulombHamiltonian,
               n_steps: int,
               control_qubit: Optional[cirq.QubitId]=None,
               omit_final_swaps: bool=False
               ) -> cirq.OP_TREE:
        # Rotate back to the computational basis
        quad_ham = QuadraticHamiltonian(hamiltonian.one_body)
        yield bogoliubov_transform(
                qubits, quad_ham.diagonalizing_bogoliubov_transform())
        # If the number of Trotter steps is odd, possibly swap qubits back
        if n_steps & 1 and not omit_final_swaps:
            yield swap_network(qubits)


class ControlledSymmetricSplitOperatorTrotterStep(
        SymmetricSplitOperatorTrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
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
        def two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
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
                               qubits: Sequence[cirq.QubitId],
                               control_qubit: Optional[cirq.QubitId]=None
                               ) -> Tuple[Sequence[cirq.QubitId],
                                          Optional[cirq.QubitId]]:
        # A Trotter step reverses the qubit ordering
        return qubits[::-1], control_qubit


SPLIT_OPERATOR = TrotterStepAlgorithm(
        supported_types={DiagonalCoulombHamiltonian},
        symmetric=SymmetricSplitOperatorTrotterStep(),
        asymmetric=None,
        controlled_symmetric=ControlledSymmetricSplitOperatorTrotterStep(),
        controlled_asymmetric=None)
