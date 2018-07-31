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

from openfermioncirq.optimization import (
        OptimizationParams,
        OptimizationResult,
        OptimizationTrialResult)
from openfermioncirq.testing import ExampleAlgorithm


def test_optimization_result_init():
    result = OptimizationResult(
            optimal_value=0.339,
            optimal_parameters=numpy.array([-1.899, -0.549]),
            num_evaluations=121,
            cost_spent=1.426,
            function_values=[(1.235, 4.119, None), (-2.452, 3.244, None)],
            wait_times=[5.329],
            time=0.423,
            seed=77,
            status=195,
            message='fdjmolGSHM')
    assert result.optimal_value == 0.339
    numpy.testing.assert_allclose(result.optimal_parameters,
                                  numpy.array([-1.899, -0.549]))
    assert result.num_evaluations == 121
    assert result.cost_spent == 1.426
    assert result.function_values == [(1.235, 4.119, None),
                                      (-2.452, 3.244, None)]
    assert result.wait_times == [5.329]
    assert result.time == 0.423
    assert result.seed == 77
    assert result.status == 195
    assert result.message == 'fdjmolGSHM'


def test_optimization_trial_result_init():
    result1 = OptimizationResult(
            optimal_value=5.7,
            optimal_parameters=numpy.array([1.3, 8.7]),
            num_evaluations=59,
            cost_spent=3.1,
            seed=60,
            status=54,
            message='ZibVTBNe8')
    result2 = OptimizationResult(
            optimal_value=4.7,
            optimal_parameters=numpy.array([1.7, 2.1]),
            num_evaluations=57,
            cost_spent=9.3,
            seed=51,
            status=32,
            message='cicCZ8iCg0D')
    trial = OptimizationTrialResult(
            [result1, result2],
            params=OptimizationParams(ExampleAlgorithm()))

    assert all(trial.data_frame['optimal_value'] == [5.7, 4.7])
    numpy.testing.assert_allclose(
            trial.data_frame['optimal_parameters'][0], numpy.array([1.3, 8.7]))
    numpy.testing.assert_allclose(
            trial.data_frame['optimal_parameters'][1], numpy.array([1.7, 2.1]))
    assert all(trial.data_frame['num_evaluations'] == [59, 57])
    assert all(trial.data_frame['cost_spent'] == [3.1, 9.3])
    assert all(trial.data_frame['seed'] == [60, 51])
    assert all(trial.data_frame['status'] == [54, 32])
    assert all(trial.data_frame['message'] == ['ZibVTBNe8', 'cicCZ8iCg0D'])


def test_optimization_trial_result_extend():
    result1 = OptimizationResult(
            optimal_value=4.7,
            optimal_parameters=numpy.array([2.3, 2.7]),
            num_evaluations=39,
            cost_spent=3.9,
            seed=63,
            status=44,
            message='di382j2f')
    result2 = OptimizationResult(
            optimal_value=3.7,
            optimal_parameters=numpy.array([1.2, 3.1]),
            num_evaluations=47,
            cost_spent=9.9,
            seed=21,
            status=22,
            message='i328d8ie3')

    trial = OptimizationTrialResult(
            [result1],
            params=OptimizationParams(ExampleAlgorithm()))

    assert len(trial.results) == 1
    assert trial.repetitions == 1

    trial.extend([result2])

    assert len(trial.results) == 2
    assert trial.repetitions == 2


def test_optimization_trial_result_data_methods():
    result1 = OptimizationResult(
            optimal_value=5.7,
            optimal_parameters=numpy.array([1.3, 8.7]),
            num_evaluations=59,
            cost_spent=3.1,
            seed=60,
            status=54,
            message='ZibVTBNe8',
            time=0.1)
    result2 = OptimizationResult(
            optimal_value=4.7,
            optimal_parameters=numpy.array([1.7, 2.1]),
            num_evaluations=57,
            cost_spent=9.3,
            seed=51,
            status=32,
            message='cicCZ8iCg0D',
            time=0.2)
    trial = OptimizationTrialResult(
            [result1, result2],
            params=OptimizationParams(ExampleAlgorithm()))

    assert trial.repetitions == 2
    assert trial.optimal_value == 4.7
    numpy.testing.assert_allclose(trial.optimal_parameters,
                                  numpy.array([1.7, 2.1]))
