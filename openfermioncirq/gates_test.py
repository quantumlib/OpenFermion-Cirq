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

from cirq.testing import EqualsTester

from openfermioncirq import LinearQubit

from openfermioncirq.gates import FSWAP, XXYY, XXYYGate


def test_fswap_interchangeable():
    a, b = LinearQubit(0), LinearQubit(1)
    assert FSWAP(a, b) == FSWAP(b, a)


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


# Waiting on Cirq Issue #242
# -------------------------
# import cirq
# def test_xxyy_on_simulator():
#
#     q0 = cirq.google.XmonQubit(0, 0)
#     q1 = cirq.google.XmonQubit(1, 0)
#     circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1), FSWAP(q0, q1),
#                                     XXYY(q0, q1) ** 0.5)
#     simulator = cirq.google.Simulator()
#     result = simulator.run(circuit)
#
#     assert cirq.allclose_up_to_global_phase(
#             result.final_states[0], numpy.array([1, -1j, -1j, 1j]) / 4)
