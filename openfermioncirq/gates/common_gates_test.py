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
from scipy.linalg import expm, kron

import cirq
import openfermioncirq as ofc


def test_fswap_interchangeable():
    a, b = cirq.LineQubit.range(2)
    assert ofc.FSWAP(a, b) == ofc.FSWAP(b, a)


def test_fswap_inverse():
    assert ofc.FSWAP**-1 == ofc.FSWAP


def test_fswap_str():
    assert str(ofc.FSWAP) == 'FSWAP'
    assert str(ofc.FSWAP**0.5) == 'FSWAP**0.5'
    assert str(ofc.FSWAP**-0.25) == 'FSWAP**-0.25'


def test_fswap_repr():
    assert repr(ofc.FSWAP) == 'ofc.FSWAP'
    assert repr(ofc.FSWAP**0.5) == '(ofc.FSWAP**0.5)'
    assert repr(ofc.FSWAP**-0.25) == '(ofc.FSWAP**-0.25)'

    ofc.testing.assert_equivalent_repr(ofc.FSWAP)
    ofc.testing.assert_equivalent_repr(ofc.FSWAP**0.5)
    ofc.testing.assert_equivalent_repr(ofc.FSWAP**-0.25)


def test_fswap_matrix():
    numpy.testing.assert_allclose(cirq.unitary(ofc.FSWAP),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, -1]]))

    numpy.testing.assert_allclose(cirq.unitary(ofc.FSWAP**0.5),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0.5+0.5j, 0.5-0.5j, 0],
                                               [0, 0.5-0.5j, 0.5+0.5j, 0],
                                               [0, 0, 0, 1j]]))

    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        val=ofc.FSWAP,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])


def test_xxyy_init():
    assert ofc.XXYYGate(half_turns=0.5).half_turns == 0.5
    assert ofc.XXYYGate(half_turns=1.5).half_turns == 1.5
    assert ofc.XXYYGate(half_turns=5).half_turns == 1


def test_xxyy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.XXYYGate(half_turns=1.0, duration=numpy.pi/2)


def test_xxyy_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(ofc.XXYYGate(half_turns=3.5),
                          ofc.XXYYGate(half_turns=-0.5),
                          ofc.XXYYGate(rads=-0.5 * numpy.pi),
                          ofc.XXYYGate(degs=-90),
                          ofc.XXYYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ofc.XXYYGate(half_turns=1.5),
                          ofc.XXYYGate(half_turns=-2.5),
                          ofc.XXYYGate(rads=1.5 * numpy.pi),
                          ofc.XXYYGate(degs=-450),
                          ofc.XXYYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ofc.XXYYGate(half_turns=0))
    eq.make_equality_group(lambda: ofc.XXYYGate(half_turns=0.5))


def test_xxyy_interchangeable():
    a, b = cirq.LineQubit(0), cirq.LineQubit(1)
    assert ofc.XXYY(a, b) == ofc.XXYY(b, a)


def test_xxyy_repr():
    assert repr(ofc.XXYYGate(half_turns=1)) == 'XXYY'
    assert repr(ofc.XXYYGate(half_turns=0.5)) == 'XXYY**0.5'


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_xxyy_decompose(half_turns):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.XXYY**half_turns)


def test_xxyy_matrix():
    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        ofc.XXYY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])

    numpy.testing.assert_allclose(cirq.unitary(ofc.XXYYGate(half_turns=2)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.XXYYGate(half_turns=1)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1j, 0],
                                               [0, -1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.XXYYGate(half_turns=0)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.XXYYGate(half_turns=-1)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    numpy.testing.assert_allclose(cirq.unitary(ofc.XXYYGate(half_turns=0.25)),
                                  expm(-1j * numpy.pi * 0.25 * (XX + YY) / 4))


def test_yxxy_init():
    assert ofc.YXXYGate(half_turns=0.5).half_turns == 0.5
    assert ofc.YXXYGate(half_turns=1.5).half_turns == 1.5
    assert ofc.YXXYGate(half_turns=5).half_turns == 1


def test_yxxy_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.YXXYGate(half_turns=1.0, duration=numpy.pi/2)


def test_yxxy_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(ofc.YXXYGate(half_turns=3.5),
                          ofc.YXXYGate(half_turns=-0.5),
                          ofc.YXXYGate(rads=-0.5 * numpy.pi),
                          ofc.YXXYGate(degs=-90),
                          ofc.YXXYGate(duration=-0.5 * numpy.pi / 2))

    eq.add_equality_group(ofc.YXXYGate(half_turns=1.5),
                          ofc.YXXYGate(half_turns=-2.5),
                          ofc.YXXYGate(rads=1.5 * numpy.pi),
                          ofc.YXXYGate(degs=-450),
                          ofc.YXXYGate(duration=-2.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ofc.YXXYGate(half_turns=0))
    eq.make_equality_group(lambda: ofc.YXXYGate(half_turns=0.5))


def test_yxxy_repr():
    assert repr(ofc.YXXYGate(half_turns=1)) == 'YXXY'
    assert repr(ofc.YXXYGate(half_turns=0.5)) == 'YXXY**0.5'


@pytest.mark.parametrize('half_turns', [1.0, 0.5, 0.25, 0.1, 0.0, -0.5])
def test_yxxy_decompose(half_turns):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
            ofc.YXXY**half_turns)


def test_yxxy_matrix():
    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        ofc.YXXY,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])


    numpy.testing.assert_allclose(cirq.unitary(ofc.YXXYGate(half_turns=2)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.YXXYGate(half_turns=1)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.YXXYGate(half_turns=0)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.YXXYGate(half_turns=-1)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    YX = kron(Y, X)
    XY = kron(X, Y)
    numpy.testing.assert_allclose(cirq.unitary(ofc.YXXYGate(half_turns=0.25)),
                                  expm(-1j * numpy.pi * 0.25 * (YX - XY) / 4))


def test_zz_init():
    assert ofc.ZZGate(half_turns=0.5).half_turns == 0.5
    assert ofc.ZZGate(half_turns=1.5).half_turns == -0.5
    assert ofc.ZZGate(half_turns=5).half_turns == 1


def test_zz_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.ZZGate(half_turns=1.0, duration=numpy.pi/2)


def test_zz_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(ofc.ZZGate(half_turns=3.5),
                          ofc.ZZGate(half_turns=-0.5),
                          ofc.ZZGate(rads=-0.5 * numpy.pi),
                          ofc.ZZGate(degs=-90),
                          ofc.ZZGate(duration=-numpy.pi / 4))

    eq.add_equality_group(ofc.ZZGate(half_turns=2.5),
                          ofc.ZZGate(half_turns=0.5),
                          ofc.ZZGate(rads=0.5 * numpy.pi),
                          ofc.ZZGate(degs=90),
                          ofc.ZZGate(duration=0.5 * numpy.pi / 2))

    eq.make_equality_group(lambda: ofc.ZZGate(half_turns=0))
    eq.make_equality_group(lambda: ofc.ZZGate(half_turns=0.1))


def test_zz_repr():
    assert repr(ofc.ZZGate(half_turns=1)) == 'ZZ'
    assert repr(ofc.ZZGate(half_turns=0.5)) == 'ZZ**0.5'


def test_zz_matrix():
    cirq.testing.assert_apply_unitary_to_tensor_is_consistent_with_unitary(
        ofc.ZZ,
        exponents=[1, -0.5, 0.5, 0.25, -0.25, 0.1, cirq.Symbol('s')])

    numpy.testing.assert_allclose(cirq.unitary(ofc.ZZGate(half_turns=0)),
                                  numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.ZZGate(half_turns=0.5)),
                                  numpy.array([[(-1j)**0.5, 0, 0, 0],
                                               [0, 1j**0.5, 0, 0],
                                               [0, 0, 1j**0.5, 0],
                                               [0, 0, 0, (-1j)**0.5]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.ZZGate(half_turns=1)),
                                  numpy.array([[-1j, 0, 0, 0],
                                               [0, 1j, 0, 0],
                                               [0, 0, 1j, 0],
                                               [0, 0, 0, -1j]]),
                                  atol=1e-8)

    numpy.testing.assert_allclose(cirq.unitary(ofc.ZZGate(half_turns=-0.5)),
                                  numpy.array([[(1j)**0.5, 0, 0, 0],
                                               [0, (-1j)**0.5, 0, 0],
                                               [0, 0, (-1j)**0.5, 0],
                                               [0, 0, 0, (1j)**0.5]]),
                                  atol=1e-8)

    Z = numpy.array([[1, 0], [0, -1]])
    ZZ = kron(Z, Z)
    numpy.testing.assert_allclose(cirq.unitary(ofc.ZZGate(half_turns=0.25)),
                                  expm(-1j * numpy.pi * 0.25 * ZZ / 2))


@pytest.mark.parametrize(
        'gate, half_turns, initial_state, correct_state, atol', [
            (ofc.XXYY, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1j, -1j, 0]) / numpy.sqrt(2), 1e-7),

            (ofc.XXYY, 0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                  numpy.array([1 / numpy.sqrt(2), 0.5, -0.5j, 0]), 1e-7),

            (ofc.XXYY, -0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1 / numpy.sqrt(2), 0.5, 0.5j, 0]), 1e-7),

            (ofc.YXXY, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, -1, 0]) / numpy.sqrt(2), 1e-7),

            (ofc.YXXY, 0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 0, 1, 0]), 1e-7),

            (ofc.YXXY, -0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                   numpy.array([0, 1, 0, 0]), 1e-7),

            (ofc.ZZ, 1.0, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, -1, -1, 0]) / numpy.sqrt(2), 1e-7),

            (ofc.ZZ, 0.5, numpy.array([0, 1, 1, 0]) / numpy.sqrt(2),
                  numpy.array([0, 1, 1, 0]) / numpy.sqrt(2), 1e-7),

            (ofc.ZZ, -0.5, numpy.array([1, 1, 0, 0]) / numpy.sqrt(2),
                   numpy.array([1, -1j, 0, 0]) / numpy.sqrt(2), 1e-7)
])
def test_two_qubit_rotation_gates_on_simulator(
        gate, half_turns, initial_state, correct_state, atol):
    simulator = cirq.google.XmonSimulator()
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(gate(a, b)**half_turns)
    initial_state = initial_state.astype(numpy.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
            result.final_state, correct_state, atol=atol)


def test_common_gate_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    circuit = cirq.Circuit.from_ops(
        ofc.FSWAP(a, b),
        ofc.FSWAP(a, b)**0.5,
        ofc.XXYY(a, b),
        ofc.YXXY(a, b),
        ofc.ZZ(a, b))
    cirq.testing.assert_has_diagram(circuit, """
a: ───×ᶠ───×ᶠ───────XXYY───YXXY───Z───
      │    │        │      │      │
b: ───×ᶠ───×ᶠ^0.5───XXYY───#2─────Z───
""")

    assert circuit.to_text_diagram(use_unicode_characters=False).strip() == """
a: ---fswap---fswap-------XXYY---YXXY---Z---
      |       |           |      |      |
b: ---fswap---fswap^0.5---XXYY---#2-----Z---
""".strip()

    circuit = cirq.Circuit.from_ops(
        ofc.XXYY(a, b)**0.5,
        ofc.YXXY(a, b)**0.5,
        ofc.ZZ(a, b)**0.5)
    cirq.testing.assert_has_diagram(circuit, """
a: ───XXYY───────YXXY─────Z───────
      │          │        │
b: ───XXYY^0.5───#2^0.5───Z^0.5───
""")
