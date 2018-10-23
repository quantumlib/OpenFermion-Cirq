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

from openfermioncirq import (
        CCZ, CXXYY, CYXXY, ControlledXXYYGate, ControlledYXXYGate, Rot111Gate)


def test_ccz_repr():
    assert repr(Rot111Gate(half_turns=1)) == 'CCZ'
    assert repr(Rot111Gate(half_turns=0.5)) == 'CCZ**0.5'


def test_cxxyy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ControlledXXYYGate(half_turns=1.0, duration=numpy.pi/2)


def test_cxxyy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ControlledXXYYGate(half_turns=3.5),
                          ControlledXXYYGate(half_turns=-0.5),
                          ControlledXXYYGate(rads=-0.5 * numpy.pi),
                          ControlledXXYYGate(degs=-90),
                          ControlledXXYYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ControlledXXYYGate(half_turns=1.5),
                          ControlledXXYYGate(half_turns=-2.5),
                          ControlledXXYYGate(rads=1.5 * numpy.pi),
                          ControlledXXYYGate(degs=-450),
                          ControlledXXYYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ControlledXXYYGate(half_turns=0))
    eq.make_equality_group(lambda: ControlledXXYYGate(half_turns=0.5))


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(half_turns):

    gate = CXXYY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, cirq.unitary(gate), atol=1e-7)


def test_cxxyy_repr():
    assert repr(ControlledXXYYGate(half_turns=1)) == 'CXXYY'
    assert repr(ControlledXXYYGate(half_turns=0.5)) == 'CXXYY**0.5'


def test_cyxxy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ControlledYXXYGate(half_turns=1.0, duration=numpy.pi/2)


def test_cyxxy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ControlledYXXYGate(half_turns=3.5),
                          ControlledYXXYGate(half_turns=-0.5),
                          ControlledYXXYGate(rads=-0.5 * numpy.pi),
                          ControlledYXXYGate(degs=-90),
                          ControlledYXXYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ControlledYXXYGate(half_turns=1.5),
                          ControlledYXXYGate(half_turns=-2.5),
                          ControlledYXXYGate(rads=1.5 * numpy.pi),
                          ControlledYXXYGate(degs=-450),
                          ControlledYXXYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ControlledYXXYGate(half_turns=0))
    eq.make_equality_group(lambda: ControlledYXXYGate(half_turns=0.5))


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(half_turns):

    gate = CYXXY**half_turns
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate.default_decompose(qubits))
    matrix = circuit.to_unitary_matrix(qubit_order=qubits)
    cirq.testing.assert_allclose_up_to_global_phase(
            matrix, cirq.unitary(gate), atol=1e-7)


def test_cyxxy_repr():
    assert repr(ControlledYXXYGate(half_turns=1)) == 'CYXXY'
    assert repr(ControlledYXXYGate(half_turns=0.5)) == 'CYXXY**0.5'


@pytest.mark.parametrize(
        'gate, half_turns, initial_state, correct_state, atol', [
            (CCZ, 0.5,
                numpy.array([1, 0, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                numpy.array([1, 0, 0, 0, 0, 0, 0, 1j]) / numpy.sqrt(2), 1e-7),
            (CCZ, 1.0,
                numpy.array([0, 1, 0, 0, 0, 0, 0, 1]) / numpy.sqrt(2),
                numpy.array([0, 1, 0, 0, 0, 0, 0, -1]) / numpy.sqrt(2), 5e-7),
            (CCZ, 0.5,
                numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 1, 0, 0, 0, 1, 0]) / numpy.sqrt(2), 5e-7),
            (CCZ, 0.25,
                numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 1, 0, 1, 0, 0]) / numpy.sqrt(2), 5e-7),
            (CCZ, -0.5,
                numpy.array([0, 0, 0, 0, 1, 0, 0, 1j]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 1, 0, 0, 1]) / numpy.sqrt(2), 1e-7),
            (CXXYY, 1.0,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / numpy.sqrt(2),
                5e-6),
            (CXXYY, 0.5,
                numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, -0.5j, 0]),
                5e-6),
            (CXXYY, -0.5,
                numpy.array([0, 0, 0, 0, 1, 1, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 1 / numpy.sqrt(2), 0.5, 0.5j, 0]),
                5e-6),
            (CXXYY, 1.0,
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0]),
                5e-6),
            (CXXYY, 1.0,
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CXXYY, 0.5,
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CXXYY, -0.5,
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CYXXY, 1.0,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 1, -1, 0]) / numpy.sqrt(2),
                5e-6),
            (CYXXY, 0.5,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 0, 1, 0]),
                5e-6),
            (CYXXY, -0.5,
                numpy.array([0, 0, 0, 0, 0, 1, 1, 0]) / numpy.sqrt(2),
                numpy.array([0, 0, 0, 0, 0, 1, 0, 0]),
                5e-6),
            (CYXXY, -0.5,
                numpy.array([1 / numpy.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                numpy.array([1, 0, 0, 0, 0, 1, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CYXXY, 1.0,
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([0, 1, 1, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CYXXY, 0.5,
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 1, 0, 0, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6),
            (CYXXY, -0.5,
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                numpy.array([1, 0, 0, 1, 0, 0, 0, 0]) / numpy.sqrt(2),
                5e-6)
])
def test_three_qubit_rotation_gates_on_simulator(
        gate, half_turns, initial_state, correct_state, atol):

    simulator = cirq.google.XmonSimulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(gate(a, b, c)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_three_qubit_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.Circuit.from_ops(
        CCZ(a, b, c),
        CXXYY(a, b, c),
        CYXXY(a, b, c))
    assert circuit.to_text_diagram().strip() == """
a: ───@───@──────@──────
      │   │      │
b: ───@───XXYY───YXXY───
      │   │      │
c: ───@───XXYY───#2─────
""".strip()

    circuit = cirq.Circuit.from_ops(
        CCZ(a, b, c)**-0.5,
        CXXYY(a, b, c)**-0.5,
        CYXXY(a, b, c)**-0.5)
    assert circuit.to_text_diagram().strip() == """
a: ───@────────@───────────@─────────
      │        │           │
b: ───@────────XXYY────────YXXY──────
      │        │           │
c: ───@^-0.5───XXYY^-0.5───#2^-0.5───
""".strip()
