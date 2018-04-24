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
from scipy.linalg import expm, kron

import cirq
from cirq.testing import EqualsTester

from openfermioncirq import LinearQubit

from openfermioncirq.gates import FSWAP, XXYY, XXYYGate, YXXY, YXXYGate


def test_fswap_interchangeable():
    a, b = LinearQubit(0), LinearQubit(1)
    assert FSWAP(a, b) == FSWAP(b, a)


def test_fswap_on_simulator():
    simulator = cirq.google.Simulator()
    a, b = cirq.google.XmonQubit(0, 0), cirq.google.XmonQubit(1, 0)
    circuit = cirq.Circuit.from_ops(FSWAP(a, b))

    initial_state = (numpy.array([1, 1, 0, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, qubits=[a, b], initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([1, 0, 1, 0]) / numpy.sqrt(2),
            atol=1e-7)

    initial_state = (numpy.array([0, 1, 0, 1], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, qubits=[a, b], initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, 0, 1, -1]) / numpy.sqrt(2),
            atol=1e-7)


def test_xx_yy_init():
    assert XXYYGate(half_turns=0.5).half_turns == 0.5
    assert XXYYGate(half_turns=5).half_turns == 1


def test_xx_yy_eq():
    eq = EqualsTester()
    eq.add_equality_group(XXYYGate(half_turns=3.5),
                          XXYYGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: XXYYGate(half_turns=0))
    eq.make_equality_pair(lambda: XXYYGate(half_turns=0.5))


def test_xx_yy_interchangeable():
    a, b = LinearQubit(0), LinearQubit(1)
    assert XXYY(a, b) == XXYY(b, a)


def test_xx_yy_extrapolate():
    assert XXYYGate(
        half_turns=1).extrapolate_effect(0.5) == XXYYGate(half_turns=0.5)


def test_xx_yy__matrix():
    assert numpy.allclose(XXYYGate(half_turns=1).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(XXYYGate(half_turns=0.5).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 0, -1j, 0],
                                       [0, -1j, 0, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(XXYYGate(half_turns=0).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(XXYYGate(half_turns=-0.5).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 0, 1j, 0],
                                       [0, 1j, 0, 0],
                                       [0, 0, 0, 1]]))

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    XX = kron(X, X)
    YY = kron(Y, Y)
    assert numpy.allclose(XXYYGate(half_turns=0.25).matrix(),
                          expm(-1j * numpy.pi * 0.25 * (XX + YY) / 2))


def test_xxyy_on_simulator():
    n_qubits = 2
    qubits = [cirq.google.XmonQubit(i, 0) for i in range(n_qubits)]
    simulator = cirq.google.Simulator()

    circuit = cirq.Circuit.from_ops(XXYY(qubits[0], qubits[1]) ** 0.5)
    initial_state = (numpy.array([0, 1, 1, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, -1j, -1j, 0]) / numpy.sqrt(2),
            atol=1e-7)

    circuit = cirq.Circuit.from_ops(XXYY(qubits[0], qubits[1]) ** .25)
    initial_state = (numpy.array([1, 1, 0, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([1, 1 / numpy.sqrt(2),
                         -1j / numpy.sqrt(2), 0]) / numpy.sqrt(2),
            atol=1e-7)

    circuit = cirq.Circuit.from_ops(XXYY(qubits[0], qubits[1]) ** -.25)
    initial_state = (numpy.array([1, 1, 0, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([1, 1 / numpy.sqrt(2),
                         1j / numpy.sqrt(2), 0]) / numpy.sqrt(2),
            atol=1e-7)


def test_yx_xy_init():
    assert YXXYGate(half_turns=0.5).half_turns == 0.5
    assert YXXYGate(half_turns=5).half_turns == 1


def test_yx_xy_eq():
    eq = EqualsTester()
    eq.add_equality_group(YXXYGate(half_turns=3.5),
                          YXXYGate(half_turns=-0.5))
    eq.make_equality_pair(lambda: YXXYGate(half_turns=0))
    eq.make_equality_pair(lambda: YXXYGate(half_turns=0.5))


def test_yx_xy_extrapolate():
    assert YXXYGate(
        half_turns=1).extrapolate_effect(0.5) == YXXYGate(half_turns=0.5)


def test_yx_xy__matrix():
    assert numpy.allclose(YXXYGate(half_turns=1).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(YXXYGate(half_turns=0.5).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(YXXYGate(half_turns=0).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))

    assert numpy.allclose(YXXYGate(half_turns=-0.5).matrix(),
                          numpy.array([[1, 0, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]]))

    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    YX = kron(X, Y)
    XY = kron(Y, X)
    assert numpy.allclose(YXXYGate(half_turns=0.25).matrix(),
                          expm(-1j * numpy.pi * 0.25 * (YX - XY) / 2))


def test_yxxy_on_simulator():
    n_qubits = 2
    qubits = [cirq.google.XmonQubit(i, 0) for i in range(n_qubits)]
    initial_state = (numpy.array([0, 1, 1, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
    simulator = cirq.google.Simulator()

    circuit = cirq.Circuit.from_ops(YXXY(qubits[0], qubits[1]) ** 0.5)
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, -1, 1, 0]) / numpy.sqrt(2),
            atol=1e-7)

    circuit = cirq.Circuit.from_ops(YXXY(qubits[0], qubits[1]) ** .25)
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, 1, 0, 0]),
            atol=1e-7)

    circuit = cirq.Circuit.from_ops(YXXY(qubits[0], qubits[1]) ** -.25)
    result = simulator.run(circuit, qubits=qubits, initial_state=initial_state)
    assert cirq.allclose_up_to_global_phase(
            result.final_states[0],
            numpy.array([0, 0, 1, 0]),
            atol=1e-7)
