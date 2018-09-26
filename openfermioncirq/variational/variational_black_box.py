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

"""Black boxes for variational studies"""

from typing import Optional, Sequence, Tuple, Union

import numpy

import cirq
from cirq import abc

from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.objective import VariationalObjective
from openfermioncirq.optimization import (
        BlackBox,
        StatefulBlackBox)


class VariationalBlackBox(BlackBox):
    """A black box encapsulating a variational ansatz objective function.

    Attributes:
        ansatz: The variational ansatz circuit.
        objective: The objective function.
        preparation_circuit: An optional circuit used to prepare the
            initial state
    """

    def __init__(self,
                 ansatz: VariationalAnsatz,
                 objective: VariationalObjective,
                 preparation_circuit: Optional[cirq.Circuit]=None,
                 initial_state: Union[int, numpy.ndarray]=0,
                 **kwargs) -> None:
        self.ansatz = ansatz
        self.objective = objective
        self.preparation_circuit = preparation_circuit or cirq.Circuit()
        self.initial_state = initial_state
        super().__init__(**kwargs)

    @property
    def dimension(self) -> int:
        """The dimension of the array accepted by the objective function."""
        return len(list(self.ansatz.params()))

    @property
    def bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """Optional bounds on the inputs to the objective function."""
        return self.ansatz.param_bounds()

    @abc.abstractmethod
    def evaluate_noiseless(self,
                           x: numpy.ndarray) -> float:
        """Evaluate parameters with a noiseless simulation."""
        pass

    def _evaluate(self,
                  x: numpy.ndarray) -> float:
        """Determine the value of some parameters."""
        # Default: defer to evaluate_noiseless
        return self.evaluate_noiseless(x)

    def _evaluate_with_cost(self,
                            x: numpy.ndarray,
                            cost: float) -> float:
        """Evaluate parameters with a specified cost."""
        # Default: add artifical noise with the specified cost
        return self._evaluate(x) + self.objective.noise(cost)

    def noise_bounds(self,
                     cost: float,
                     confidence: Optional[float]=None
                     ) -> Tuple[float, float]:
        """Exact or approximate bounds on noise in the objective function."""
        return self.objective.noise_bounds(cost, confidence)


class UnitarySimulateVariationalBlackBox(VariationalBlackBox):

    def evaluate_noiseless(self,
                           x: numpy.ndarray) -> float:
        """Evaluate parameters with a noiseless simulation."""
        # Default: evaluate using apply_unitary_effect_to_state
        circuit = (self.preparation_circuit + self.ansatz.circuit
                  ).with_parameters_resolved_by(self.ansatz.param_resolver(x))
        final_state = circuit.apply_unitary_effect_to_state(
                self.initial_state,
                qubit_order=self.ansatz.qubit_permutation(self.ansatz.qubits))
        return self.objective.value(final_state)


class UnitarySimulateVariationalStatefulBlackBox(
        UnitarySimulateVariationalBlackBox,
        StatefulBlackBox):
    """A stateful black box encapsulating a variational objective function."""
    pass


class XmonSimulateVariationalBlackBox(VariationalBlackBox):

    def evaluate_noiseless(self,
                           x: numpy.ndarray) -> float:
        """Evaluate parameters with a noiseless simulation."""
        # Default: evaluate using Xmon simulator
        simulator = cirq.google.XmonSimulator()
        result = simulator.simulate(
                self.preparation_circuit + self.ansatz.circuit,
                initial_state=self.initial_state,
                param_resolver=self.ansatz.param_resolver(x),
                qubit_order=self.ansatz.qubit_permutation(self.ansatz.qubits))
        return self.objective.value(result)


class XmonSimulateVariationalStatefulBlackBox(XmonSimulateVariationalBlackBox,
                                              StatefulBlackBox):
    """A stateful black box encapsulating a variational objective function."""
    pass


UNITARY_SIMULATE = UnitarySimulateVariationalBlackBox
UNITARY_SIMULATE_STATEFUL = UnitarySimulateVariationalStatefulBlackBox
XMON_SIMULATE = XmonSimulateVariationalBlackBox
XMON_SIMULATE_STATEFUL = XmonSimulateVariationalStatefulBlackBox
