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

from typing import Optional, Sequence, Union, cast

import os

import numpy
import pytest

import cirq

from openfermioncirq import (
        OptimizationParams,
        VariationalAnsatz,
        VariationalStudy)
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult,
        OptimizationTrialResult,
        ScipyOptimizationAlgorithm)
from openfermioncirq.variational.study import VariationalStudyBlackBox


class ExampleAnsatz(VariationalAnsatz):

    def param_names(self) -> Sequence[str]:
        return ['theta{}'.format(i) for i in range(2)]

    def _generate_qubits(self) -> Sequence[cirq.QubitId]:
        return cirq.LineQubit.range(2)

    def operations(self, qubits: Sequence[cirq.QubitId]) -> cirq.OP_TREE:
        a, b = qubits
        yield cirq.RotXGate(half_turns=self.params['theta0']).on(a)
        yield cirq.RotXGate(half_turns=self.params['theta1']).on(b)
        yield cirq.CZ(a, b)
        yield cirq.RotXGate(half_turns=self.params['theta0']).on(a)
        yield cirq.RotXGate(half_turns=self.params['theta1']).on(b)
        yield cirq.MeasurementGate('all').on(a, b)


class ExampleStudy(VariationalStudy):

    def value(self,
              trial_result: Union[cirq.TrialResult,
                                  cirq.google.XmonSimulateTrialResult]
              ) -> float:
        measurements = trial_result.measurements['all']
        return numpy.sum(measurements)


class ExampleStudyNoisy(ExampleStudy):

    def noise(self, cost: Optional[float]=None) -> float:
        return numpy.exp(-cast(float, cost))


class ExampleAlgorithm(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        if initial_guess is None:
            # coverage: ignore
            initial_guess = numpy.ones(black_box.dimension)
        if initial_guess_array is None:
            # coverage: ignore
            initial_guess_array = numpy.ones((3, black_box.dimension))
        a = black_box.evaluate(initial_guess)
        b = black_box.evaluate_with_cost(initial_guess_array[0], 1.0)
        return OptimizationResult(optimal_value=min(a, b),
                                  optimal_parameters=initial_guess,
                                  num_evaluations=1,
                                  cost_spent=0.0,
                                  status=0,
                                  message='success')


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
    numpy.testing.assert_allclose(test_study_noisy.noise(2.0), numpy.exp(-2.0))


def test_variational_study_evaluate():
    numpy.testing.assert_allclose(
            test_study.evaluate(test_study.default_initial_params()), 1)


def test_variational_study_evaluate_with_cost():
    numpy.testing.assert_allclose(
            test_study.evaluate_with_cost(
                test_study.default_initial_params(), 2.0),
            1)
    numpy.testing.assert_allclose(
            test_study_noisy.evaluate_with_cost(
                test_study.default_initial_params(), 2.0),
            1 + numpy.exp(-2.0))


def test_variational_study_noise_bounds():
    assert test_study.noise_bounds(100) == (-numpy.inf, numpy.inf)


def test_variational_study_optimize_and_summary():
    numpy.random.seed(63351)

    study = ExampleStudy('study', test_ansatz)
    assert len(study.results) == 0

    result = study.optimize(
            OptimizationParams(test_algorithm),
            'run1')
    assert len(study.results) == 1
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1

    study.optimize(OptimizationParams(test_algorithm),
                   repetitions=2,
                   use_multiprocessing=True)
    result = study.results[0]
    assert len(study.results) == 2
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 2

    study.optimize(
            OptimizationParams(
                test_algorithm,
                initial_guess=numpy.array([4.5, 8.8]),
                initial_guess_array=numpy.array([[7.2, 6.3],
                                                 [3.6, 9.8]]),
                cost_of_evaluate=1.0),
            reevaluate_final_params=True)
    result = study.results[1]
    assert len(study.results) == 3
    assert isinstance(result, OptimizationTrialResult)
    assert result.repetitions == 1
    assert all(result.data_frame['optimal_parameters'].apply(study.evaluate) ==
               result.data_frame['optimal_value'])
    numpy.testing.assert_allclose(result.params.initial_guess,
                                  numpy.array([4.5, 8.8]))
    numpy.testing.assert_allclose(result.params.initial_guess_array,
                                  numpy.array([[7.2, 6.3],
                                               [3.6, 9.8]]))
    assert result.params.cost_of_evaluate == 1.0

    assert study.summary.replace("u'", "'").strip() == """
This study contains 3 results.
The optimal value found among all results is 0.
It was found by the run with identifier 'run1'.
Result details:
    Identifier: run1
        Optimal value: 0
        Number of repetitions: 1
        Optimal value 1st, 2nd, 3rd quartiles:
            [0.0, 0.0, 0.0]
        Num evaluations 1st, 2nd, 3rd quartiles:
            [2.0, 2.0, 2.0]
        Cost spent 1st, 2nd, 3rd quartiles:
            [1.0, 1.0, 1.0]
    Identifier: 0
        Optimal value: 0
        Number of repetitions: 2
        Optimal value 1st, 2nd, 3rd quartiles:
            [0.0, 0.0, 0.0]
        Num evaluations 1st, 2nd, 3rd quartiles:
            [2.0, 2.0, 2.0]
        Cost spent 1st, 2nd, 3rd quartiles:
            [1.0, 1.0, 1.0]
    Identifier: 1
        Optimal value: 0
        Number of repetitions: 1
        Optimal value 1st, 2nd, 3rd quartiles:
            [0.0, 0.0, 0.0]
        Num evaluations 1st, 2nd, 3rd quartiles:
            [2.0, 2.0, 2.0]
        Cost spent 1st, 2nd, 3rd quartiles:
            [2.0, 2.0, 2.0]
""".strip()


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
                    options={'maxiter': 1}),
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
    assert result.params.algorithm.options == {'maxiter': 1}
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
