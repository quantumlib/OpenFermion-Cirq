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

from openfermioncirq.optimization.scipy import (
        COBYLA,
        L_BFGS_B,
        NELDER_MEAD,
        SLSQP,
        ScipyOptimizationAlgorithm)
from openfermioncirq.testing import ExampleBlackBox


@pytest.mark.parametrize('algorithm', [COBYLA, L_BFGS_B, NELDER_MEAD, SLSQP])
def test_scipy_algorithm(algorithm):
    black_box = ExampleBlackBox()
    initial_guess = numpy.zeros(black_box.dimension)
    result = algorithm.optimize(black_box, initial_guess)

    assert isinstance(result.optimal_value, float)
    assert isinstance(result.optimal_parameters, numpy.ndarray)
    assert isinstance(result.num_evaluations, int)
    assert isinstance(result.status, int)
    assert isinstance(result.message, (str, bytes))


def test_scipy_algorithm_requires_initial_guess():
    black_box = ExampleBlackBox()
    with pytest.raises(ValueError):
        _ = COBYLA.optimize(black_box)

def test_scipy_algorithm_name():
    assert ScipyOptimizationAlgorithm().name == 'ScipyOptimizationAlgorithm'
    assert COBYLA.name == 'COBYLA'
    assert L_BFGS_B.name == 'L-BFGS-B'
    assert NELDER_MEAD.name == 'Nelder-Mead'
    assert SLSQP.name == 'SLSQP'
