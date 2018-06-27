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

from typing import Optional

import numpy
import pytest

from openfermioncirq.optimization import BlackBox, OptimizationResult

from openfermioncirq.optimization.algorithm import OptimizationAlgorithm


class ExampleBlackBox(BlackBox):

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self,
                 x: numpy.ndarray) -> float:
        return numpy.sum(x**2)


class ExampleAlgorithm(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        if initial_guess is None:
            initial_guess = numpy.ones(black_box.dimension)
        if initial_guess_array is None:
            initial_guess_array = numpy.ones(3 * black_box.dimension).reshape(
                    (3, black_box.dimension))
        a = black_box.evaluate(initial_guess)
        b = black_box.evaluate(initial_guess_array[0])
        c = black_box.evaluate(initial_guess_array[1])
        d = black_box.evaluate(initial_guess_array[2])
        return OptimizationResult(optimal_value=min(a, b, c, d),
                                  optimal_parameters=initial_guess,
                                  num_evaluations=1,
                                  cost_spent=0.0,
                                  status=0,
                                  message='success')


def test_optimization_algorithm_options():
    algorithm = ExampleAlgorithm()
    assert algorithm.options == {}


def test_optimization_algorithm_optimize():
    black_box = ExampleBlackBox()
    algorithm = ExampleAlgorithm()
    result = algorithm.optimize(black_box)

    numpy.testing.assert_allclose(result.optimal_parameters, numpy.ones(2))
    assert result.optimal_value == 2.0


def test_optimization_algorithm_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = OptimizationAlgorithm()


def test_optimization_algorithm_is_abstract_must_implement():
    class Missing(OptimizationAlgorithm):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_optimization_algorithm_is_abstract_can_implement():
    class Included(OptimizationAlgorithm):
        def optimize(self):
            pass

    assert isinstance(Included(), OptimizationAlgorithm)
