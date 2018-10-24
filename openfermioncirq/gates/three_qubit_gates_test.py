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
from cirq.testing import EqualsTester

import openfermioncirq as ofc


def test_cxxyy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.ControlledXXYYGate(half_turns=1.0, duration=numpy.pi/2)


def test_apply_unitary_effect():
    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        ofc.CXXYY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])

    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        ofc.CYXXY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])


def test_cxxyy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CXXYY**-0.5,
                          ofc.ControlledXXYYGate(half_turns=3.5),
                          ofc.ControlledXXYYGate(half_turns=-0.5),
                          ofc.ControlledXXYYGate(rads=-0.5 * numpy.pi),
                          ofc.ControlledXXYYGate(degs=-90),
                          ofc.ControlledXXYYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ofc.ControlledXXYYGate(half_turns=1.5),
                          ofc.ControlledXXYYGate(half_turns=-2.5),
                          ofc.ControlledXXYYGate(rads=1.5 * numpy.pi),
                          ofc.ControlledXXYYGate(degs=-450),
                          ofc.ControlledXXYYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ofc.ControlledXXYYGate(half_turns=0))
    eq.make_equality_group(lambda: ofc.ControlledXXYYGate(half_turns=0.5))


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(half_turns):

    gate = ofc.CXXYY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, cirq.unitary(gate), atol=1e-7)


def test_cxxyy_repr():
    assert repr(ofc.CXXYY) == 'CXXYY'
    assert repr(ofc.CXXYY**0.5) == 'CXXYY**0.5'


def test_cyxxy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.ControlledYXXYGate(half_turns=1.0, duration=numpy.pi/2)


def test_cyxxy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CYXXY**-0.5,
                          ofc.ControlledYXXYGate(half_turns=3.5),
                          ofc.ControlledYXXYGate(half_turns=-0.5),
                          ofc.ControlledYXXYGate(rads=-0.5 * numpy.pi),
                          ofc.ControlledYXXYGate(degs=-90),
                          ofc.ControlledYXXYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ofc.ControlledYXXYGate(half_turns=1.5),
                          ofc.ControlledYXXYGate(half_turns=-2.5),
                          ofc.ControlledYXXYGate(rads=1.5 * numpy.pi),
                          ofc.ControlledYXXYGate(degs=-450),
                          ofc.ControlledYXXYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ofc.ControlledYXXYGate(half_turns=0))
    eq.make_equality_group(lambda: ofc.ControlledYXXYGate(half_turns=0.5))


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(half_turns):

    gate = ofc.CYXXY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, cirq.unitary(gate), atol=1e-7)


def test_cyxxy_repr():
    assert repr(ofc.ControlledYXXYGate(half_turns=1)) == 'CYXXY'
    assert repr(ofc.ControlledYXXYGate(half_turns=0.5)) == 'CYXXY**0.5'


@pytest.mark.parametrize(
        'gate, initial_state, correct_state', [
            (ofc.CXXYY,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / numpy.sqrt(2)),
            (ofc.CXXYY**0.5,
                numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, -0.5j, 0])),
            (ofc.CXXYY**-0.5,
                numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, 0.5j, 0])),
            (ofc.CXXYY,
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0])),
            (ofc.CXXYY,
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2)),
            (ofc.CXXYY**0.5,
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2)),
            (ofc.CXXYY**-0.5,
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2)),
            (ofc.CYXXY,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 1, -1, 0]) / numpy.sqrt(2)),
            (ofc.CYXXY**0.5,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 0, 1, 0])),
            (ofc.CYXXY**-0.5,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 1, 0, 0])),
            (ofc.CYXXY**-0.5,
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                numpy.array([1, 0, 0, 0, 0, 1, 0, 0]) / numpy.sqrt(2)),
            (ofc.CYXXY,
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2)),
            (ofc.CYXXY**0.5,
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2)),
            (ofc.CYXXY**-0.5,
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2))
])
def test_three_qubit_rotation_gates_on_simulator(gate: cirq.Gate,
                                                 initial_state: numpy.ndarray,
                                                 correct_state: numpy.ndarray):
    op = gate(*cirq.LineQubit.range(3))
    result = cirq.Circuit.from_ops(op).apply_unitary_effect_to_state(
        initial_state, dtype=numpy.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(result,
                                                    correct_state,
                                                    atol=1e-8)


def test_three_qubit_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c),
        ofc.CYXXY(a, b, c))
    assert circuit.to_text_diagram().strip() == """
a: ───@──────@──────
      │      │
b: ───XXYY───YXXY───
      │      │
c: ───XXYY───#2─────
""".strip()

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c)**-0.5,
        ofc.CYXXY(a, b, c)**-0.5)
    assert circuit.to_text_diagram().strip() == """
a: ───@───────────@─────────
      │           │
b: ───XXYY────────YXXY──────
      │           │
c: ───XXYY^-0.5───#2^-0.5───
""".strip()
