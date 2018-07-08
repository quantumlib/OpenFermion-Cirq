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

from typing import Optional, Sequence, cast

import os

import numpy
import pytest

import cirq

from openfermioncirq import VariationalAnsatz
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)

from openfermioncirq.variational.study import (
        VariationalStudy,
        VariationalStudyBlackBox)


class ExampleAnsatz(VariationalAnsatz):

    def param_names(self) -> Sequence[str]:
        return ['theta{}'.format(i) for i in range(8)]

    def _generate_qubits(self) -> Sequence[cirq.QubitId]:
        return cirq.LineQubit.range(4)

    def _generate_circuit(self, qubits: Sequence[cirq.QubitId]) -> cirq.Circuit:
        a, b, c, d = qubits
        return cirq.Circuit.from_ops(
                cirq.RotXGate(half_turns=self.params['theta0']).on(a),
                cirq.RotXGate(half_turns=self.params['theta1']).on(b),
                cirq.RotXGate(half_turns=self.params['theta2']).on(c),
                cirq.RotXGate(half_turns=self.params['theta3']).on(d),
                cirq.CNOT(a, b),
                cirq.CNOT(c, d),
                cirq.CNOT(b, c),
                cirq.RotZGate(half_turns=self.params['theta4']).on(a),
                cirq.RotZGate(half_turns=self.params['theta5']).on(b),
                cirq.RotZGate(half_turns=self.params['theta6']).on(c),
                cirq.RotZGate(half_turns=self.params['theta7']).on(d),
                cirq.MeasurementGate('all').on(a, b, c, d))


class ExampleStudy(VariationalStudy):

    def value(self,
              trial_result: cirq.TrialResult) -> float:
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

a, b, _, _ = test_ansatz.qubits
preparation_circuit = cirq.Circuit.from_ops(cirq.X(a), cirq.X(b))
test_study = ExampleStudy('test_study',
                          test_ansatz,
                          preparation_circuit=preparation_circuit)
test_study_noisy = ExampleStudyNoisy('test_study_noisy',
                                     test_ansatz,
                                     preparation_circuit=preparation_circuit)

test_algorithm = ExampleAlgorithm()


def test_variational_study_circuit():
    assert (test_study.circuit.to_text_diagram().strip() == """
0: ───X───X^theta0───@───Z^theta4──────────────M───
                     │                         │
1: ───X───X^theta1───X───@──────────Z^theta5───M───
                         │                     │
2: ───────X^theta2───@───X──────────Z^theta6───M───
                     │                         │
3: ───────X^theta3───X──────────────Z^theta7───M───
""".strip())


def test_variational_study_num_params():
    assert test_study.num_params == 8


def test_variational_study_ansatz_properties():
    assert test_study.qubits == test_ansatz.qubits
    assert test_study.param_names() == test_ansatz.param_names()
    assert test_study.param_bounds() == test_ansatz.param_bounds()
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


def test_variational_study_run():
    study = ExampleStudy('study', test_ansatz,)
    assert len(study.results) == 0

    study.run('run1',
              test_algorithm)
    assert len(study.results) == 1
    assert isinstance(study.results['run1'], list)
    assert len(study.results['run1']) == 1
    assert isinstance(study.results['run1'][0], OptimizationResult)

    study.run('run2',
              test_algorithm,
              cost_of_evaluate=1.0)
    assert len(study.results) == 2
    assert isinstance(study.results['run2'], list)
    assert len(study.results['run2']) == 1
    assert isinstance(study.results['run1'][0], OptimizationResult)

    study.run('run3',
              test_algorithm,
              repetitions=2,
              use_multiprocessing=True,
              num_processes=2)
    assert len(study.results) == 3
    assert isinstance(study.results['run2'], list)
    assert len(study.results['run3']) == 2
    assert isinstance(study.results['run3'][1], OptimizationResult)


def test_variational_study_run_too_few_seeds_raises_error():
    with pytest.raises(ValueError):
        test_study.run('run',
                       test_algorithm,
                       repetitions=2,
                       seeds=[0])


def test_variational_study_save_load():
    datadir = 'tmp_yulXPXnMBrxeUVt7kYVw'
    study_name = 'test_study'

    study = ExampleStudy(
            study_name,
            test_ansatz,
            datadir=datadir)
    study.run('run',
              test_algorithm,
              repetitions=2)
    study.save()

    loaded_study = VariationalStudy.load(study_name, datadir=datadir)

    assert loaded_study.name == study.name
    assert str(loaded_study.circuit) == str(study.circuit)
    assert len(loaded_study.results) == 1
    assert len(loaded_study.results['run']) == 2
    assert isinstance(loaded_study.results['run'][0], OptimizationResult)
    assert loaded_study.datadir == datadir

    loaded_study = VariationalStudy.load('{}.study'.format(study_name),
                                         datadir=datadir)

    assert loaded_study.name == study.name
    assert str(loaded_study.circuit) == str(study.circuit)
    assert len(loaded_study.results) == 1
    assert len(loaded_study.results['run']) == 2
    assert isinstance(loaded_study.results['run'][0], OptimizationResult)
    assert loaded_study.datadir == datadir

    # Clean up
    os.remove(os.path.join(datadir, '{}.study'.format(study_name)))
    os.rmdir(datadir)


def test_variational_study_black_box_dimension():
    black_box = VariationalStudyBlackBox(test_study)
    assert black_box.dimension == len(test_study.param_names())


def test_variational_study_black_box_bounds():
    black_box = VariationalStudyBlackBox(test_study)
    assert black_box.bounds == test_study.param_bounds()
