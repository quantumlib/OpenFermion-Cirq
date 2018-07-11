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

from openfermioncirq.optimization.black_box import BlackBox


class ExampleBlackBox(BlackBox):

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self,
                 x: numpy.ndarray) -> float:
        return numpy.sum(x**2)


class ExampleBlackBoxNoisy(ExampleBlackBox):

    def evaluate_with_cost(self,
                           x: numpy.ndarray,
                           cost: float) -> float:
        return numpy.sum(x**2) + 1 / cost


def test_black_box_dimension():
    black_box = ExampleBlackBox()
    assert black_box.dimension == 2


def test_black_box_evaluate():
    black_box = ExampleBlackBox()
    assert black_box.evaluate(numpy.array([1.0, 2.0])) == 5.0


def test_black_box_evaluate_with_cost():
    black_box = ExampleBlackBox()
    black_box_noisy = ExampleBlackBoxNoisy()
    assert black_box.evaluate_with_cost(numpy.array([1.0, 2.0]), 1.0) == 5.0
    assert black_box_noisy.evaluate_with_cost(
            numpy.array([1.0, 2.0]), 2.0) == 5.5


def test_black_box_noise_bounds():
    black_box = ExampleBlackBox()
    assert black_box.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_black_box_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = BlackBox()


def test_black_box_is_abstract_must_implement():
    class Missing1(BlackBox):
        @property
        def dimension(self):
            pass
    class Missing2(BlackBox):
        def evaluate(self):
            pass

    with pytest.raises(TypeError):
        _ = Missing1()
    with pytest.raises(TypeError):
        _ = Missing2()


def test_black_box_is_abstract_can_implement():
    class Included(BlackBox):
        @property
        def dimension(self):
            pass
        def evaluate(self):
            pass

    assert isinstance(Included(), BlackBox)
