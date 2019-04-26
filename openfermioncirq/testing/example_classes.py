# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subclasses of abstract classes for use in tests."""

from typing import Iterable, Optional, Sequence, Union, cast

import numpy
import sympy

import cirq

from openfermioncirq.optimization.algorithm import OptimizationAlgorithm
from openfermioncirq.optimization.black_box import BlackBox, StatefulBlackBox
from openfermioncirq.optimization.result import OptimizationResult
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.objective import VariationalObjective


class ExampleAlgorithm(OptimizationAlgorithm):
    """Evaluates 5 random points and returns the best answer found."""

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        opt = numpy.inf
        opt_params = None
        for _ in range(5):
            guess = numpy.random.randn(black_box.dimension)
            val = black_box.evaluate(guess)
            if val < opt:
                opt = val
                opt_params = guess
        return OptimizationResult(
                optimal_value=opt,
                optimal_parameters=cast(numpy.ndarray, opt_params),
                num_evaluations=1,
                cost_spent=0.0,
                status=0,
                message='success')


class LazyAlgorithm(OptimizationAlgorithm):
    """Just returns the initial guess, or the zeros vector."""

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        if initial_guess is None:
            # coverage: ignore
            initial_guess = numpy.zeros(black_box.dimension)
        opt = black_box.evaluate(initial_guess)
        return OptimizationResult(
                optimal_value=opt,
                optimal_parameters=initial_guess,
                num_evaluations=1,
                cost_spent=0.0,
                status=0,
                message='success')


class ExampleBlackBox(BlackBox):
    """Returns the sum of the squares of the inputs."""

    @property
    def dimension(self) -> int:
        return 2

    def _evaluate(self,
                  x: numpy.ndarray) -> float:
        return numpy.sum(x**2)


class ExampleBlackBoxNoisy(ExampleBlackBox):
    """Returns the sum of the squares of the inputs plus some noise.
    The noise is drawn from the standard normal distribution, then divided
    by the cost provided.
    """

    def _evaluate_with_cost(self,
                            x: numpy.ndarray,
                            cost: float) -> float:
        return numpy.sum(x**2) + numpy.random.randn() / cost


class ExampleStatefulBlackBox(ExampleBlackBox, StatefulBlackBox):
    """Returns the sum of the squares of the inputs."""
    pass


class ExampleAnsatz(VariationalAnsatz):
    """An example variational ansatz.

    The ansatz produces the operations::

        0: ───X^theta0───@───X^theta0───M('all')───
                         │              │
        1: ───X^theta1───@───X^theta1───M──────────
    """

    def params(self) -> Iterable[sympy.Symbol]:
        for i in range(2):
            yield sympy.Symbol('theta{}'.format(i))

    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        return cirq.LineQubit.range(2)

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        a, b = qubits
        yield cirq.XPowGate(exponent=sympy.Symbol('theta0')).on(a)
        yield cirq.XPowGate(exponent=sympy.Symbol('theta1')).on(b)
        yield cirq.CZ(a, b)
        yield cirq.XPowGate(exponent=sympy.Symbol('theta0')).on(a)
        yield cirq.XPowGate(exponent=sympy.Symbol('theta1')).on(b)
        yield cirq.measure(a, b, key='all')


class ExampleVariationalObjective(VariationalObjective):
    """An example variational objective.

    The value of the study is the number of qubits that were measured to be 1.
    """

    def value(self,
              circuit_output: Union[cirq.TrialResult,
                                    cirq.SimulationTrialResult,
                                    numpy.ndarray]
              ) -> float:
        if isinstance(circuit_output, numpy.ndarray):
            return sum(bin(i).count('1') * p for i, p in
                    enumerate(circuit_output))
        measurements = circuit_output.measurements['all']
        return numpy.sum(measurements)


class ExampleVariationalObjectiveNoisy(ExampleVariationalObjective):
    """An example variational objective with a noise model.

    The noise is drawn from the standard normal distribution, then divided
    by the cost provided. If a cost is not specified, the noise is 0.
    """

    def noise(self, cost: Optional[float]=None) -> float:
        if cost is None:
            return 0.0  # coverage: ignore
        return numpy.random.randn() / cost
