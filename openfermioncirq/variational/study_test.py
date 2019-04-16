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

from openfermioncirq import VariationalObjective, VariationalStudy
from openfermioncirq.optimization import (
        OptimizationParams,
        OptimizationTrialResult,
        ScipyOptimizationAlgorithm)
from openfermioncirq.variational import variational_black_box
from openfermioncirq.variational.study import (
        VariationalStudy)
from openfermioncirq.variational.variational_black_box import (
        UnitarySimulateVariationalBlackBox,
        XmonSimulateVariationalBlackBox)
from openfermioncirq.testing import (
        ExampleAlgorithm,
        ExampleAnsatz,
        ExampleVariationalObjective,
        ExampleVariationalObjectiveNoisy,
        LazyAlgorithm)


test_algorithm = ExampleAlgorithm()
test_ansatz = ExampleAnsatz()
test_objective = ExampleVariationalObjective()
test_objective_noisy = ExampleVariationalObjectiveNoisy()

a, b = test_ansatz.qubits
preparation_circuit = cirq.Circuit.from_ops(cirq.X(a))
test_study = VariationalStudy(
        'test_study',
        test_ansatz,
        test_objective,
        preparation_circuit=preparation_circuit,
        black_box_type=variational_black_box.XMON_SIMULATE)
test_study_noisy = VariationalStudy(
        'test_study_noisy',
        test_ansatz,
        test_objective_noisy,
        preparation_circuit=preparation_circuit,
        black_box_type=variational_black_box.XMON_SIMULATE)


def test_variational_study_circuit():
    assert (test_study.circuit.to_text_diagram().strip() == """
0: ───X───X^theta0───@───X^theta0───M('all')───
                     │              │
1: ───────X^theta1───@───X^theta1───M──────────
""".strip())


def test_variational_study_num_params():
    assert test_study.num_params == 2


def test_variational_study_optimize_and_extend_and_summary():
    numpy.random.seed(63351)

    study = VariationalStudy(
            'study', test_ansatz, test_objective,
            black_box_type=variational_black_box.UNITARY_SIMULATE_STATEFUL,
            target=-10.5)
    assert len(study.trial_results) == 0
    assert study.target == -10.5

    # Optimization run 1
    result = study.optimize(
            OptimizationParams(test_algorithm),
            'run1')
    assert len(study.trial_results) == 1
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1

    # Extend optimization run 1
    study.extend_result('run1',
                        repetitions=2)
    assert study.trial_results['run1'].repetitions == 3

    # Optimization run 2
    study.optimize(OptimizationParams(test_algorithm),
                   repetitions=2,
                   use_multiprocessing=True)
    result = study.trial_results[0]
    assert len(study.trial_results) == 2
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
            save_x_vals=True)
    result = study.trial_results[1]
    assert len(study.trial_results) == 3
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1
    assert all(
            result.data_frame['optimal_parameters'].apply(
                lambda x: UnitarySimulateVariationalBlackBox(
                    test_ansatz, test_objective).evaluate(x))
            == result.data_frame['optimal_value'])
    assert isinstance(result.results[0].cost_spent, float)

    # Optimization run 4
    trial_results = study.optimize_sweep(
            [OptimizationParams(test_algorithm),
                OptimizationParams(LazyAlgorithm())],
            identifiers=['test', 'lazy'],
            use_multiprocessing=True,
            num_processes=2
    )
    assert isinstance(trial_results[0].params.algorithm, ExampleAlgorithm)
    assert isinstance(trial_results[1].params.algorithm, LazyAlgorithm)
    assert isinstance(study.trial_results['test'].params.algorithm,
                      ExampleAlgorithm)
    assert isinstance(study.trial_results['lazy'].params.algorithm,
                      LazyAlgorithm)

    # Try extending non-existent run
    with pytest.raises(KeyError):
        study.extend_result('run100')

    # Check that getting a summary works
    assert str(study).startswith('This study contains')


def test_variational_study_initial_state():
    preparation_circuit = cirq.Circuit.from_ops(cirq.X(test_ansatz.qubits[0]))
    initial_state = numpy.array([0.0, 0.0, 1.0, 0.0])

    class TestObjective(VariationalObjective):
        def value(self, circuit_output):
            return circuit_output[0].real

    study1 = VariationalStudy(
            'study1',
            test_ansatz,
            TestObjective(),
            preparation_circuit=preparation_circuit,
            black_box_type=variational_black_box.UNITARY_SIMULATE_STATEFUL)
    study2 = VariationalStudy(
            'study2',
            test_ansatz,
            TestObjective(),
            initial_state=initial_state,
            black_box_type=variational_black_box.UNITARY_SIMULATE_STATEFUL)

    initial_guess = numpy.random.randn(2)
    result1 = study1.optimize(
            OptimizationParams(
                LazyAlgorithm(), initial_guess=initial_guess))
    result2 = study2.optimize(
            OptimizationParams(
                LazyAlgorithm(), initial_guess=initial_guess))

    numpy.testing.assert_allclose(result1.optimal_value, result2.optimal_value)


def test_variational_study_run_too_few_seeds_raises_error():
    with pytest.raises(ValueError):
        test_study.optimize(OptimizationParams(test_algorithm),
                            'run',
                            repetitions=2,
                            seeds=[0])


def test_variational_study_save_load():
    datadir = 'tmp_yulXPXnMBrxeUVt7kYVw'
    study_name = 'test_study'

    study = VariationalStudy(
            study_name,
            test_ansatz,
            test_objective,
            initial_state=numpy.array([0.0, 1.0, 0.0, 0.0]).astype(
                numpy.complex64),
            datadir=datadir,
            black_box_type=variational_black_box.UNITARY_SIMULATE_STATEFUL)
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
    assert len(loaded_study.trial_results) == 1
    numpy.testing.assert_allclose(loaded_study.initial_state,
                                  numpy.array([0.0, 1.0, 0.0, 0.0]))

    result = loaded_study.trial_results['example']
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


def test_variational_black_box_dimension():
    black_box = XmonSimulateVariationalBlackBox(test_ansatz, test_objective)
    assert black_box.dimension == 2


def test_variational_black_box_bounds():
    black_box = XmonSimulateVariationalBlackBox(test_ansatz, test_objective)
    assert black_box.bounds == test_study.ansatz.param_bounds()


def test_variational_black_box_noise_bounds():
    black_box = XmonSimulateVariationalBlackBox(test_ansatz, test_objective)
    assert black_box.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_variational_black_box_evaluate():
    black_box = UnitarySimulateVariationalBlackBox(test_ansatz, test_objective)
    numpy.testing.assert_allclose(
            black_box.evaluate(test_ansatz.default_initial_params()), 0.0)
    numpy.testing.assert_allclose(
            black_box.evaluate(numpy.array([0.0, 0.5])), 1.0)


def test_variational_black_box_evaluate_with_cost():
    black_box = UnitarySimulateVariationalBlackBox(test_ansatz, test_objective)

    numpy.testing.assert_allclose(
            black_box.evaluate_with_cost(
                test_ansatz.default_initial_params(), 2.0),
            0.0)

    black_box_noisy = UnitarySimulateVariationalBlackBox(
            test_ansatz, test_objective_noisy)
    numpy.random.seed(33534)
    noisy_val = black_box_noisy.evaluate_with_cost(
            numpy.array([0.5, 0.0]), 10.0)
    assert -0.8 < noisy_val < 1.2
