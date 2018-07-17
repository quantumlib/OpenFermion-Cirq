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

from typing import Sequence

import numpy
import pytest

import cirq

from openfermioncirq.variational.ansatz import VariationalAnsatz


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


def test_variational_ansatz_circuit():
    ansatz = ExampleAnsatz()
    assert ansatz.circuit.to_text_diagram().strip() == """
0: ───X^theta0───@───X^theta0───M('all')───
                 │              │
1: ───X^theta1───@───X^theta1───M──────────
""".strip()


def test_variational_ansatz_param_bounds():
    ansatz = ExampleAnsatz()
    assert ansatz.param_bounds() is None


def test_variational_ansatz_param_resolver():
    ansatz = ExampleAnsatz()
    resolver = ansatz.param_resolver(numpy.arange(2, dtype=float))
    assert resolver['theta0'] == 0
    assert resolver['theta1'] == 1


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
        def _generate_qubits(self):
            pass
    class Missing2(VariationalAnsatz):
        def _generate_qubits(self):
            pass
        def operations(self, qubits):
            pass
    class Missing3(VariationalAnsatz):
        def operations(self, qubits):
            pass
        def param_names(self):
            return []  # coverage: ignore

    with pytest.raises(TypeError):
        _ = Missing1()
    with pytest.raises(TypeError):
        _ = Missing2()
    with pytest.raises(TypeError):
        _ = Missing3()


def test_variational_ansatz_is_abstract_can_implement():
    class Included(VariationalAnsatz):
        def param_names(self):
            return []  # coverage: ignore
        def _generate_qubits(self):
            pass
        def operations(self, qubits):
            return ()

    assert isinstance(Included(), VariationalAnsatz)
