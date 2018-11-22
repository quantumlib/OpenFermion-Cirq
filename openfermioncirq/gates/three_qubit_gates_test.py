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

import numpy as np
import pytest

import cirq
from cirq.testing import EqualsTester

import openfermioncirq as ofc


def test_cxxyy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.ControlledXXYYGate(exponent=1.0, duration=np.pi/2)


def test_apply_unitary_effect():
    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.CXXYY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])

    cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
        ofc.CYXXY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])


def test_cxxyy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CXXYY**-0.5,
                          ofc.ControlledXXYYGate(exponent=3.5),
                          ofc.ControlledXXYYGate(exponent=-0.5),
                          ofc.ControlledXXYYGate(rads=-0.5 * np.pi),
                          ofc.ControlledXXYYGate(degs=-90),
                          ofc.ControlledXXYYGate(duration=-0.5 * np.pi / 2))

    eq.add_equality_group(ofc.ControlledXXYYGate(exponent=1.5),
                          ofc.ControlledXXYYGate(exponent=-2.5),
                          ofc.ControlledXXYYGate(rads=1.5 * np.pi),
                          ofc.ControlledXXYYGate(degs=-450),
                          ofc.ControlledXXYYGate(duration=-2.5 * np.pi / 2))

    eq.make_equality_group(lambda: ofc.ControlledXXYYGate(exponent=0))
    eq.make_equality_group(lambda: ofc.ControlledXXYYGate(exponent=0.5))


@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cxxyy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.CXXYY**exponent)


def test_cxxyy_repr():
    assert repr(ofc.CXXYY) == 'CXXYY'
    assert repr(ofc.CXXYY**0.5) == 'CXXYY**0.5'


def test_cyxxy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.ControlledYXXYGate(exponent=1.0, duration=np.pi/2)


def test_cyxxy_eq():
    eq = EqualsTester()

    eq.add_equality_group(ofc.CYXXY**-0.5,
                          ofc.ControlledYXXYGate(exponent=3.5),
                          ofc.ControlledYXXYGate(exponent=-0.5),
                          ofc.ControlledYXXYGate(rads=-0.5 * np.pi),
                          ofc.ControlledYXXYGate(degs=-90),
                          ofc.ControlledYXXYGate(duration=-0.5 * np.pi / 2))

    eq.add_equality_group(ofc.ControlledYXXYGate(exponent=1.5),
                          ofc.ControlledYXXYGate(exponent=-2.5),
                          ofc.ControlledYXXYGate(rads=1.5 * np.pi),
                          ofc.ControlledYXXYGate(degs=-450),
                          ofc.ControlledYXXYGate(duration=-2.5 * np.pi / 2))

    eq.make_equality_group(lambda: ofc.ControlledYXXYGate(exponent=0))
    eq.make_equality_group(lambda: ofc.ControlledYXXYGate(exponent=0.5))


@pytest.mark.parametrize('exponent', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_cyxxy_decompose(exponent):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.CYXXY**exponent)


def test_cyxxy_repr():
    assert repr(ofc.ControlledYXXYGate(exponent=1)) == 'CYXXY'
    assert repr(ofc.ControlledYXXYGate(exponent=0.5)) == 'CYXXY**0.5'


@pytest.mark.parametrize(
        'gate, initial_state, correct_state', [
            (ofc.CXXYY,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, -1j, -1j, 0]) / np.sqrt(2)),
            (ofc.CXXYY**0.5,
                np.array([0, 0, 0, 0, 1, 1, 0, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 1 / np.sqrt(2), 0.5, -0.5j, 0])),
            (ofc.CXXYY**-0.5,
                np.array([0, 0, 0, 0, 1, 1, 0, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 1 / np.sqrt(2), 0.5, 0.5j, 0])),
            (ofc.CXXYY,
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, -0.5j, -0.5j, 0])),
            (ofc.CXXYY,
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CXXYY**0.5,
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CXXYY**-0.5,
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 1, -1, 0]) / np.sqrt(2)),
            (ofc.CYXXY**0.5,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 0, 1, 0])),
            (ofc.CYXXY**-0.5,
                np.array([0, 0, 0, 0, 0, 1, 1, 0]) / np.sqrt(2),
                np.array([0, 0, 0, 0, 0, 1, 0, 0])),
            (ofc.CYXXY**-0.5,
                np.array([1 / np.sqrt(2), 0, 0, 0, 0, 0.5, 0.5, 0]),
                np.array([1, 0, 0, 0, 0, 1, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY,
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY**0.5,
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2)),
            (ofc.CYXXY**-0.5,
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2),
                np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2))
])
def test_three_qubit_rotation_gates_on_simulator(gate: cirq.Gate,
                                                 initial_state: np.ndarray,
                                                 correct_state: np.ndarray):
    op = gate(*cirq.LineQubit.range(3))
    result = cirq.Circuit.from_ops(op).apply_unitary_effect_to_state(
        initial_state, dtype=np.complex128)
    cirq.testing.assert_allclose_up_to_global_phase(result,
                                                    correct_state,
                                                    atol=1e-8)


@pytest.mark.parametrize('rads', [
    2*np.pi, np.pi, 0.5*np.pi, 0.25*np.pi, 0.1*np.pi, 0.0, -0.5*np.pi])
def test_crxxyy_unitary(rads):
    np.testing.assert_allclose(
            cirq.unitary(ofc.CRxxyy(rads)),
            cirq.unitary(cirq.ControlledGate(ofc.Rxxyy(rads))),
            atol=1e-8)


@pytest.mark.parametrize('rads', [
    2*np.pi, np.pi, 0.5*np.pi, 0.25*np.pi, 0.1*np.pi, 0.0, -0.5*np.pi])
def test_cryxxy_unitary(rads):
    np.testing.assert_allclose(
            cirq.unitary(ofc.CRyxxy(rads)),
            cirq.unitary(cirq.ControlledGate(ofc.Ryxxy(rads))),
            atol=1e-8)


def test_three_qubit_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c),
        ofc.CYXXY(a, b, c))
    cirq.testing.assert_has_diagram(circuit, """
a: ───@──────@──────
      │      │
b: ───XXYY───YXXY───
      │      │
c: ───XXYY───#2─────
""")

    circuit = cirq.Circuit.from_ops(
        ofc.CXXYY(a, b, c)**-0.5,
        ofc.CYXXY(a, b, c)**-0.5)
    cirq.testing.assert_has_diagram(circuit, """
a: ───@───────────@─────────
      │           │
b: ───XXYY────────YXXY──────
      │           │
c: ───XXYY^-0.5───#2^-0.5───
""")
