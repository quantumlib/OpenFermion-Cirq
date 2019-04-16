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

import cirq

from openfermioncirq import VariationalObjective
from openfermioncirq.testing import (
        ExampleVariationalObjective,
        ExampleVariationalObjectiveNoisy)


test_objective = ExampleVariationalObjective()
test_objective_noisy = ExampleVariationalObjectiveNoisy()


def test_variational_objective_value():
    simulator = cirq.Simulator()
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(
            cirq.X.on_each(*qubits[:3]),
            cirq.measure(*qubits, key='all'))
    result = simulator.simulate(circuit)

    numpy.testing.assert_allclose(test_objective.value(result), 3)


def test_variational_objective_noise():
    numpy.testing.assert_allclose(test_objective.noise(2.0), 0.0)

    numpy.random.seed(26347)
    assert -0.6 < test_objective_noisy.noise(2.0) < 0.6


def test_variational_objective_noise_bounds():
    assert test_objective.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_variational_objective_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = VariationalObjective()


def test_variational_objective_is_abstract_must_implement():
    class Missing(VariationalObjective):
        pass
    with pytest.raises(TypeError):
        _ = Missing()


def test_variational_objective_is_abstract_can_implement():
    class Included(VariationalObjective):
        def value(self):
            pass
    assert isinstance(Included(), VariationalObjective)
