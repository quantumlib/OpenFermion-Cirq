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

import os

import numpy
import pytest

import cirq

from openfermioncirq import VariationalStudy
from openfermioncirq.optimization import (
        OptimizationParams,
        OptimizationTrialResult,
        ScipyOptimizationAlgorithm,
        StatefulBlackBox)
from openfermioncirq.variational.study import VariationalStudyBlackBox
from openfermioncirq.testing import (
        ExampleAlgorithm,
        ExampleAnsatz,
        ExampleStudy,
        ExampleStudyNoisy)


test_ansatz = ExampleAnsatz()

a, b = test_ansatz.qubits
preparation_circuit = cirq.Circuit.from_ops(cirq.X(a))
test_study = ExampleStudy('test_study',
                          test_ansatz,
                          preparation_circuit=preparation_circuit)
test_study_noisy = ExampleStudyNoisy('test_study_noisy',
                                     test_ansatz,
                                     preparation_circuit=preparation_circuit)

test_algorithm = ExampleAlgorithm()


def test_variational_study_circuit():
    assert (test_study.circuit.to_text_diagram().strip() == """
0: ───X───X^theta0───@───X^theta0───M('all')───
                     │              │
1: ───────X^theta1───@───X^theta1───M──────────
""".strip())


def test_variational_study_num_params():
    assert test_study.num_params == 2


def test_variational_study_ansatz_properties():
    numpy.testing.assert_allclose(test_study.default_initial_params(),
                                  test_ansatz.default_initial_params())


def test_variational_study_value():
    simulator = cirq.google.XmonSimulator()
    result = simulator.simulate(
            test_study.circuit,
            param_resolver=test_ansatz.param_resolver(numpy.ones(8)))

    numpy.testing.assert_allclose(test_study.value(result), 1)


def test_variational_study_noise():
    numpy.testing.assert_allclose(test_study.noise(2.0), 0.0)

    numpy.random.seed(26347)
    assert -0.6 < test_study_noisy.noise(2.0) < 0.6


def test_variational_study_evaluate():
    numpy.testing.assert_allclose(
            test_study.evaluate(test_study.default_initial_params()), 1)


def test_variational_study_evaluate_with_cost():
    numpy.testing.assert_allclose(
            test_study.evaluate_with_cost(
                test_study.default_initial_params(), 2.0),
            1)

    numpy.random.seed(33534)
    noisy_val = test_study_noisy.evaluate_with_cost(
            test_study.default_initial_params(), 10.0)
    assert 0.8 < noisy_val < 1.2


def test_variational_study_noise_bounds():
    assert test_study.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_variational_study_optimize_and_summary():
    numpy.random.seed(63351)

    study = ExampleStudy('study', test_ansatz)
    assert len(study.results) == 0

    # Optimization run 1
    result = study.optimize(
            OptimizationParams(test_algorithm),
            'run1')
    assert len(study.results) == 1
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1

    # Optimization run 2
    study.optimize(OptimizationParams(test_algorithm),
                   repetitions=2,
                   use_multiprocessing=True)
    result = study.results[0]
    assert len(study.results) == 2
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 2

    # Optimization run 3
    study.optimize(
            OptimizationParams(
                test_algorithm,
                initial_guess=numpy.array([4.5, 8.8]),
                initial_guess_array=numpy.array([[7.2, 6.3],
                                                 [3.6, 9.8]]),
                cost_of_evaluate=1.0),
            reevaluate_final_params=True,
            stateful=True,
            save_x_vals=True)
    result = study.results[1]
    assert len(study.results) == 3
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1
    assert all(result.data_frame['optimal_parameters'].apply(study.evaluate) ==
               result.data_frame['optimal_value'])
    assert isinstance(result.results[0].black_box, StatefulBlackBox)

    # Check that getting a summary works
    assert isinstance(study.summary, str)


def test_variational_study_run_too_few_seeds_raises_error():
    with pytest.raises(ValueError):
        test_study.optimize(OptimizationParams(test_algorithm),
                            'run',
                            repetitions=2,
                            seeds=[0])


def test_variational_study_save_load():
    datadir = 'tmp_yulXPXnMBrxeUVt7kYVw'
    study_name = 'test_study'

    study = ExampleStudy(
            study_name,
            test_ansatz,
            datadir=datadir)
    study.optimize(
            OptimizationParams(
                ScipyOptimizationAlgorithm(
                    kwargs={'method': 'COBYLA'},
                    options={'maxiter': 2}),
                initial_guess=numpy.array([7.9, 3.9]),
                initial_guess_array=numpy.array([[7.5, 7.6],
                                                 [8.8, 1.1]]),
                cost_of_evaluate=1.0),
            'example')
    study.save()

    loaded_study = VariationalStudy.load(study_name, datadir=datadir)

    assert loaded_study.name == study.name
    assert str(loaded_study.circuit) == str(study.circuit)
    assert loaded_study.datadir == datadir
    assert len(loaded_study.results) == 1

    result = loaded_study.results['example']
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1
    assert isinstance(result.params.algorithm, ScipyOptimizationAlgorithm)
    assert result.params.algorithm.kwargs == {'method': 'COBYLA'}
    assert result.params.algorithm.options == {'maxiter': 2}
    assert result.params.cost_of_evaluate == 1.0

    loaded_study = VariationalStudy.load('{}.study'.format(study_name),
                                         datadir=datadir)

    assert loaded_study.name == study.name

    # Clean up
    os.remove(os.path.join(datadir, '{}.study'.format(study_name)))
    os.rmdir(datadir)


def test_variational_study_black_box_dimension():
    black_box = VariationalStudyBlackBox(test_study)
    assert black_box.dimension == len(test_study.ansatz.param_names())


def test_variational_study_black_box_bounds():
    black_box = VariationalStudyBlackBox(test_study)
    assert black_box.bounds == test_study.ansatz.param_bounds()


def test_variational_study_black_box_noise_bounds():
    black_box = VariationalStudyBlackBox(test_study)
    assert black_box.noise_bounds(100) == (-numpy.inf, numpy.inf)
