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

from typing import List

import numpy
import pytest

import cirq

from openfermioncirq.variational.ansatz import VariationalAnsatz


class ExampleAnsatz(VariationalAnsatz):

    def __init__(self):
        self.qubits = cirq.LineQubit.range(4)
        super().__init__()

    def param_names(self) -> List[str]:
        return ['theta{}'.format(i) for i in range(8)]

    def generate_circuit(self) -> cirq.Circuit:

        a, b, c, d = self.qubits
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


def test_variational_ansatz_circuit():
    ansatz = ExampleAnsatz()
    assert ansatz.circuit.to_text_diagram().strip() == """
0: ───X^theta0───@───Z^theta4──────────────M───
                 │                         │
1: ───X^theta1───X───@──────────Z^theta5───M───
                     │                     │
2: ───X^theta2───@───X──────────Z^theta6───M───
                 │                         │
3: ───X^theta3───X──────────────Z^theta7───M───
""".strip()


def test_variational_ansatz_param_bounds():
    ansatz = ExampleAnsatz()
    assert ansatz.param_bounds() is None


def test_variational_ansatz_param_resolver():
    ansatz = ExampleAnsatz()
    resolver = ansatz.param_resolver(numpy.arange(8, dtype=float))
    assert resolver['theta0'] == 0
    assert resolver['theta1'] == 1
    assert resolver['theta2'] == 2
    assert resolver['theta3'] == 3
    assert resolver['theta4'] == 4
    assert resolver['theta5'] == 5
    assert resolver['theta6'] == 6
    assert resolver['theta7'] == 7


def test_variational_ansatz_default_initial_params():
    ansatz = ExampleAnsatz()
    numpy.testing.assert_allclose(ansatz.default_initial_params(),
                                  numpy.zeros(len(ansatz.params)))


def test_variational_ansatz_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = VariationalAnsatz()


def test_variational_ansatz_is_abstract_must_implement():
    class Missing1(VariationalAnsatz):
        def param_names(self):
            return []  # coverage: ignore
    class Missing2(VariationalAnsatz):
        def generate_circuit(self):
            pass

    with pytest.raises(TypeError):
        _ = Missing1()
    with pytest.raises(TypeError):
        _ = Missing2()


def test_variational_ansatz_is_abstract_can_implement():
    class Included(VariationalAnsatz):
        def param_names(self):
            return []  # coverage: ignore
        def generate_circuit(self):
            pass

    assert isinstance(Included(), VariationalAnsatz)
