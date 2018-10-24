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

"""A Trotter algorithm using the "fermionic simulation gate"."""

from typing import Optional, Sequence, Tuple

import cirq
from openfermion import DiagonalCoulombHamiltonian

from openfermioncirq import (
        ControlledXXYYGate,
        ControlledYXXYGate,
        rot111,
        XXYYGate,
        YXXYGate,
        swap_network)

from openfermioncirq.trotter.trotter_algorithm import (
        Hamiltonian,
        TrotterStep,
        TrotterAlgorithm)


class LinearSwapNetworkTrotterAlgorithm(TrotterAlgorithm):
    """A Trotter algorithm using the "fermionic simulation gate".

    This algorithm simulates a DiagonalCoulombHamiltonian. It uses layers of
    fermionic swap networks to simultaneously simulate the one- and two-body
    interactions.

    This algorithm is described in arXiv:1711.04789.
    """

    supported_types = {DiagonalCoulombHamiltonian}

    def symmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return SymmetricLinearSwapNetworkTrotterStep(hamiltonian)

    def asymmetric(self, hamiltonian: Hamiltonian) -> Optional[TrotterStep]:
        return AsymmetricLinearSwapNetworkTrotterStep(hamiltonian)

    def controlled_symmetric(self, hamiltonian: Hamiltonian
                             ) -> Optional[TrotterStep]:
        return ControlledSymmetricLinearSwapNetworkTrotterStep(hamiltonian)

    def controlled_asymmetric(self, hamiltonian: Hamiltonian
                              ) -> Optional[TrotterStep]:
        return ControlledAsymmetricLinearSwapNetworkTrotterStep(hamiltonian)


LINEAR_SWAP_NETWORK = LinearSwapNetworkTrotterAlgorithm()


class SymmetricLinearSwapNetworkTrotterStep(TrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Apply one- and two-body interactions for half of the full time
        def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
            yield XXYYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].real * time).on(a, b)
            yield YXXYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield cirq.Rot11Gate(rads=
                    -self.hamiltonian.two_body[p, q] * time).on(a, b)
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.RotZGate(rads=
                   -self.hamiltonian.one_body[i, i].real * time).on(qubits[i])
               for i in range(n_qubits))

        # Apply one- and two-body interactions for half of the full time
        # This time, reorder the operations so that the entire Trotter step is
        # symmetric
        def one_and_two_body_interaction_reverse_order(p, q, a, b
                ) -> cirq.OP_TREE:
            yield cirq.Rot11Gate(rads=
                    -self.hamiltonian.two_body[p, q] * time).on(a, b)
            yield YXXYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield XXYYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].real * time).on(a, b)
        yield swap_network(qubits, one_and_two_body_interaction_reverse_order,
                fermionic=True, offset=True)


class ControlledSymmetricLinearSwapNetworkTrotterStep(TrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Apply one- and two-body interactions for half of the full time
        def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
            yield ControlledXXYYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].real * time).on(
                            control_qubit, a, b)
            yield ControlledYXXYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].imag * time).on(
                            control_qubit, a, b)
            yield rot111(-self.hamiltonian.two_body[p, q] * time).on(
                            control_qubit, a, b)
        yield swap_network(
                qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.Rot11Gate(rads=
                   -self.hamiltonian.one_body[i, i].real * time).on(
                       control_qubit, qubits[i])
               for i in range(n_qubits))

        # Apply one- and two-body interactions for half of the full time
        # This time, reorder the operations so that the entire Trotter step is
        # symmetric
        def one_and_two_body_interaction_reverse_order(p, q, a, b
                ) -> cirq.OP_TREE:
            yield rot111(-self.hamiltonian.two_body[p, q] * time).on(
                control_qubit, a, b)
            yield ControlledYXXYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].imag * time).on(
                            control_qubit, a, b)
            yield ControlledXXYYGate(duration=
                    0.5 * self.hamiltonian.one_body[p, q].real * time).on(
                            control_qubit, a, b)
        yield swap_network(qubits, one_and_two_body_interaction_reverse_order,
                fermionic=True, offset=True)

        # Apply phase from constant term
        yield cirq.RotZGate(rads=
                -self.hamiltonian.constant * time).on(control_qubit)

class AsymmetricLinearSwapNetworkTrotterStep(TrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Apply one- and two-body interactions for the full time
        def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
            yield XXYYGate(duration=
                    self.hamiltonian.one_body[p, q].real * time).on(a, b)
            yield YXXYGate(duration=
                    self.hamiltonian.one_body[p, q].imag * time).on(a, b)
            yield cirq.Rot11Gate(rads=
                    -2 * self.hamiltonian.two_body[p, q] * time).on(a, b)
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.RotZGate(rads=
                   -self.hamiltonian.one_body[i, i].real * time).on(qubits[i])
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
               n_steps: int,
               control_qubit: Optional[cirq.QubitId]=None,
               omit_final_swaps: bool=False
               ) -> cirq.OP_TREE:
        # If the number of Trotter steps is odd, possibly swap qubits back
        if n_steps & 1 and not omit_final_swaps:
            yield swap_network(qubits, fermionic=True)


class ControlledAsymmetricLinearSwapNetworkTrotterStep(TrotterStep):

    def trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
            ) -> cirq.OP_TREE:

        n_qubits = len(qubits)

        # Apply one- and two-body interactions for the full time
        def one_and_two_body_interaction(p, q, a, b) -> cirq.OP_TREE:
            yield ControlledXXYYGate(duration=
                    self.hamiltonian.one_body[p, q].real * time).on(
                            control_qubit, a, b)
            yield ControlledYXXYGate(duration=
                    self.hamiltonian.one_body[p, q].imag * time).on(
                            control_qubit, a, b)
            yield rot111(-2 * self.hamiltonian.two_body[p, q] * time).on(
                control_qubit, a, b)
        yield swap_network(qubits, one_and_two_body_interaction, fermionic=True)
        qubits = qubits[::-1]

        # Apply one-body potential for the full time
        yield (cirq.Rot11Gate(rads=
                   -self.hamiltonian.one_body[i, i].real * time).on(
                       control_qubit, qubits[i])
               for i in range(n_qubits))

        # Apply phase from constant term
        yield cirq.RotZGate(rads=
                -self.hamiltonian.constant * time).on(control_qubit)

    def step_qubit_permutation(self,
                               qubits: Sequence[cirq.QubitId],
                               control_qubit: Optional[cirq.QubitId]=None
                               ) -> Tuple[Sequence[cirq.QubitId],
                                          Optional[cirq.QubitId]]:
        # A Trotter step reverses the qubit ordering
        return qubits[::-1], control_qubit

    def finish(self,
               qubits: Sequence[cirq.QubitId],
               n_steps: int,
               control_qubit: Optional[cirq.QubitId]=None,
               omit_final_swaps: bool=False
               ) -> cirq.OP_TREE:
        # If the number of Trotter steps is odd, possibly swap qubits back
        if n_steps & 1 and not omit_final_swaps:
            yield swap_network(qubits, fermionic=True)
