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

from openfermioncirq.optimization.result import OptimizationResult


def test_optimization_result_init():
    result = OptimizationResult(
            optimal_value=0.339,
            optimal_parameters=numpy.array([-1.899, -0.549]),
            num_evaluations=121,
            cost_spent=1.426,
            initial_guess=numpy.array([0.398, -1.271]),
            status=195,
            message='fdjmolGSHM')
    assert result.optimal_value == 0.339
    numpy.testing.assert_allclose(result.optimal_parameters,
                                  numpy.array([-1.899, -0.549]))
    assert result.num_evaluations == 121
    assert result.cost_spent == 1.426
    numpy.testing.assert_allclose(result.initial_guess,
                                  numpy.array([0.398, -1.271]))
    assert result.status == 195
    assert result.message == 'fdjmolGSHM'
