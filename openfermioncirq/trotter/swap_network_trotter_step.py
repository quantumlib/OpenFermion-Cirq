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

import numpy

import cirq
from openfermion import DiagonalCoulombHamiltonian

from openfermioncirq import CCZ, CXXYY, CYXXY, XXYY, YXXY, swap_network

from openfermioncirq.trotter.trotter_step_algorithm import TrotterStepAlgorithm


class SwapNetworkTrotterStep(TrotterStepAlgorithm):
    """A Trotter step using two consecutive fermionic swap networks.

    This algorithm is described in arXiv:1711.04789.
    """

    def second_order_trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
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


SWAP_NETWORK = SwapNetworkTrotterStep()


class ControlledSwapNetworkTrotterStep(TrotterStepAlgorithm):

    controlled = True

    def second_order_trotter_step(
            self,
            qubits: Sequence[cirq.QubitId],
            hamiltonian: DiagonalCoulombHamiltonian,
            time: float,
            control_qubit: Optional[cirq.QubitId]=None
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


CONTROLLED_SWAP_NETWORK = ControlledSwapNetworkTrotterStep()
