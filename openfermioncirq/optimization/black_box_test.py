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

from openfermioncirq.optimization.black_box import BlackBox, StatefulBlackBox
from openfermioncirq.testing import (
        ExampleBlackBox,
        ExampleBlackBoxNoisy,
        ExampleStatefulBlackBox)


def test_black_box_dimension():
    black_box = ExampleBlackBox()
    assert black_box.dimension == 2


def test_black_box_evaluate():
    black_box = ExampleBlackBox()
    assert black_box.evaluate(numpy.array([1.0, 2.0])) == 5.0


def test_black_box_evaluate_with_cost():
    black_box = ExampleBlackBox()
    assert black_box.evaluate_with_cost(numpy.array([1.0, 2.0]), 1.0) == 5.0

    numpy.random.seed(14536)
    black_box_noisy = ExampleBlackBoxNoisy()
    noisy_val = black_box_noisy.evaluate_with_cost(
            numpy.array([1.0, 2.0]), 10.0)
    assert 5.0 < noisy_val < 6.0


def test_black_box_noise_bounds():
    black_box = ExampleBlackBox()
    assert black_box.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_black_box_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = BlackBox()


def test_stateful_black_box():
    stateful_black_box = ExampleStatefulBlackBox()
    a, b, c, d = numpy.random.randn(4, 2)
    _ = stateful_black_box.evaluate(a)
    _ = stateful_black_box.evaluate(b)
    _ = stateful_black_box.evaluate_with_cost(c, 1.0)
    _ = stateful_black_box.evaluate_with_cost(d, 2.0)

    assert len(stateful_black_box.function_values) == 4
    assert stateful_black_box.num_evaluations == 4
    assert stateful_black_box.cost_spent == 3.0

    y, z, x = stateful_black_box.function_values[0]
    assert isinstance(y, float)
    assert z is None
    assert x is None

    y, z, x = stateful_black_box.function_values[2]
    assert isinstance(y, float)
    assert z == 1.0
    assert x is None

    assert len(stateful_black_box.wait_times) == 3
    for t in stateful_black_box.wait_times:
        assert isinstance(t, float)


def test_stateful_black_box_save_x_vals():
    stateful_black_box = ExampleStatefulBlackBox(save_x_vals=True)
    a, b, c, d = numpy.random.randn(4, 2)
    _ = stateful_black_box.evaluate(a)
    _ = stateful_black_box.evaluate(b)
    _ = stateful_black_box.evaluate_with_cost(c, 1.0)
    _ = stateful_black_box.evaluate_with_cost(d, 2.0)

    y, z, x = stateful_black_box.function_values[0]
    assert isinstance(y, float)
    assert z is None
    assert isinstance(x, numpy.ndarray)


def test_black_box_is_abstract_must_implement():
    class Missing1(BlackBox):
        @property
        def dimension(self):
            pass
    class Missing2(BlackBox):
        def _evaluate(self):
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
        def _evaluate(self):
            pass

    assert isinstance(Included(), BlackBox)


def test_stateful_black_box_is_abstract_must_implement():
    class Missing1(StatefulBlackBox):
        @property
        def dimension(self):
            pass
    class Missing2(StatefulBlackBox):
        def _evaluate(self):
            pass

    with pytest.raises(TypeError):
        _ = Missing1()
    with pytest.raises(TypeError):
        _ = Missing2()


def test_stateful_black_box_is_abstract_can_implement():
    class Included(StatefulBlackBox):
        @property
        def dimension(self):
            pass
        def _evaluate(self):
            pass

    assert isinstance(Included(), StatefulBlackBox)
