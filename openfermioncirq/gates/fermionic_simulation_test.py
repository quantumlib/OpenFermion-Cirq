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

import itertools

import numpy as np
import pytest
import scipy.linalg as la
import sympy

import cirq
import openfermioncirq as ofc
from openfermioncirq.gates.fermionic_simulation import (
        state_swap_eigen_component)


def test_state_swap_eigen_component_args():
    with pytest.raises(TypeError):
        state_swap_eigen_component(0, '12', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '01', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '10', 0)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', '100', 1)
    with pytest.raises(ValueError):
        state_swap_eigen_component('01', 'ab', 1)


@pytest.mark.parametrize('index_pair,n_qubits', [
    ((0, 1), 2),
    ((0, 3), 2),
    ])
def test_state_swap_eigen_component(index_pair, n_qubits):
    state_pair = tuple(format(i, '0' + str(n_qubits) + 'b') for i in index_pair)
    i, j = index_pair
    dim = 2 ** n_qubits
    for sign in (-1, 1):
        actual_component = state_swap_eigen_component(
                state_pair[0], state_pair[1], sign)
        expected_component = np.zeros((dim, dim))
        expected_component[i, i] = expected_component[j, j] = 0.5
        expected_component[i, j] = expected_component[j, i] = sign * 0.5
        assert np.allclose(actual_component, expected_component)


def test_quadratic_fermionic_simulation_gate():
    ofc.testing.assert_implements_consistent_protocols(
        ofc.QuadraticFermionicSimulationGate())


def test_quadratic_fermionic_simulation_gate_zero_weights():
    gate = ofc.QuadraticFermionicSimulationGate((0, 0))

    assert np.allclose(cirq.unitary(gate), np.eye(4))
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


@pytest.mark.parametrize('weights,exponent', [
    ((np.random.uniform(-5, 5) + 1j * np.random.uniform(-5, 5),
        np.random.uniform(-5, 5)), np.random.uniform(-5, 5)) for _ in range(5)
])
def test_quadratic_fermionic_simulation_gate_unitary(
        weights, exponent):
    generator = np.zeros((4, 4), dtype=np.complex128)
    # w0 |10><01| + h.c.
    generator[2, 1] = weights[0]
    generator[1, 2] = weights[0].conjugate()
    # w1 |11><11|
    generator[3, 3] = weights[1]
    expected_unitary = la.expm(-1j * exponent * generator)

    gate  = ofc.QuadraticFermionicSimulationGate(weights, exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)

    symbolic_gate = (
            ofc.QuadraticFermionicSimulationGate(
                (sympy.Symbol('w0'), sympy.Symbol('w1')),
                exponent=sympy.Symbol('t')))
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(symbolic_gate._decompose_(qubits))
    resolver = {'w0': weights[0], 'w1': weights[1], 't': exponent}
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    decomp_unitary = resolved_circuit.to_unitary_matrix(qubit_order=qubits)

    assert np.allclose(expected_unitary, decomp_unitary)

    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


def test_cubic_fermionic_simulation_gate():
    ofc.testing.assert_eigengate_implements_consistent_protocols(
        ofc.CubicFermionicSimulationGate)


def test_cubic_fermionic_simulation_gate_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate() ** 0.5,
        ofc.CubicFermionicSimulationGate((1,) * 3, exponent=0.5),
        ofc.CubicFermionicSimulationGate((0.5,) * 3)
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((1j, 0, 0)),
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((sympy.Symbol('s'), 0, 0), exponent=2),
        ofc.CubicFermionicSimulationGate(
            (2 * sympy.Symbol('s'), 0, 0), exponent=1)
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((0, 0.7, 0), global_shift=2),
        ofc.CubicFermionicSimulationGate(
            (0, 0.35, 0), global_shift=1, exponent=2)
        )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate((1, 1, 1))
    )
    eq.add_equality_group(
        ofc.CubicFermionicSimulationGate(((1 + 2 * np.pi), 1, 1))
    )


@pytest.mark.parametrize('exponent,control',
    itertools.product(
        [0, 1, -1, 0.25, -0.5, 0.1],
        [0, 1, 2]))
def test_cubic_fermionic_simulation_gate_consistency_special(exponent, control):
    weights = tuple(np.eye(1, 3, control)[0] * 0.5 * np.pi)
    general_gate  = ofc.CubicFermionicSimulationGate(weights, exponent=exponent)
    general_unitary = cirq.unitary(general_gate)

    indices = np.dot(
            list(itertools.product((0, 1), repeat=3)),
            (2 ** np.roll(np.arange(3), -control))[::-1])
    special_gate = cirq.ControlledGate(cirq.ISWAP**-exponent)
    special_unitary = (
            cirq.unitary(special_gate)[indices[:, np.newaxis], indices])

    assert np.allclose(general_unitary, special_unitary)


@pytest.mark.parametrize('weights,exponent', [
    (np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
        np.random.uniform(-5, 5)) for _ in range(5)
])
def test_cubic_fermionic_simulation_gate_consistency_docstring(
        weights, exponent):
    generator = np.zeros((8, 8), dtype=np.complex128)
    # w0 |110><101| + h.c.
    generator[6, 5] = weights[0]
    generator[5, 6] = weights[0].conjugate()
    # w1 |110><011| + h.c.
    generator[6, 3] = weights[1]
    generator[3, 6] = weights[1].conjugate()
    # w2 |101><011| + h.c.
    generator[5, 3] = weights[2]
    generator[3, 5] = weights[2].conjugate()
    expected_unitary = la.expm(-1j * exponent * generator)

    gate  = ofc.CubicFermionicSimulationGate(weights, exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)


def test_quartic_fermionic_simulation_consistency():
    ofc.testing.assert_implements_consistent_protocols(
        ofc.QuarticFermionicSimulationGate())


@pytest.mark.parametrize('weights', np.random.rand(10, 3))
def test_weights_and_exponent(weights):
    exponents = np.linspace(-1, 1, 8)
    gates = tuple(
        ofc.QuarticFermionicSimulationGate(weights / exponent,
                                         exponent=exponent)
        for exponent in exponents)

    for g1 in gates:
        for g2 in gates:
            assert cirq.approx_eq(g1, g2, atol=1e-100)

    for i, (gate, exponent) in enumerate(zip(gates, exponents)):
        assert gate.exponent == 1
        new_exponent = exponents[-i]
        new_gate = gate._with_exponent(new_exponent)
        assert new_gate.exponent == new_exponent

quartic_fermionic_simulation_simulator_test_cases = [
        (ofc.QuarticFermionicSimulationGate((0, 0, 0)), 1.,
         np.ones(16) / 4.,
         np.ones(16) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0.2, -0.1, 0.7)), 0.3,
         np.array([1, -1, -1, -1, -1, -1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]) / 4.,
         np.array([1, -1, -1, -np.exp(0.21j),
                      -1, -np.exp(-0.03j),
                      np.exp(-0.06j), 1,
                      1, np.exp(-0.06j),
                      np.exp(-0.03j), 1,
                      np.exp(0.21j), 1, 1, 1]) / 4.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((1. / 3, 0, 0)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 1., 0,
                      0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
         np.array([0, 0, 0, 0, 0, 0, 1., 0,
                      0, 1., 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, np.pi / 3, 0)), 1.,
         np.array([1., 1., 0, 0, 0, 1., 0, 0,
                      0, 0., -1., 0, 0, 0, 0, 0]) / 2.,
         np.array([1., 1., 0, 0, 0, -np.exp(4j * np.pi / 3), 0, 0,
                      0, 0., -np.exp(1j * np.pi / 3), 0, 0, 0, 0, 0]
                     ) / 2.,
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, 0, -np.pi / 2)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1., 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0]),
         5e-6),
        (ofc.QuarticFermionicSimulationGate((0, 0, -0.25 * np.pi)), 1.,
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 1j, 0, 0, 0]) / np.sqrt(2),
         5e-6),
        (ofc.QuarticFermionicSimulationGate(
            (-np.pi / 4, np.pi /6, -np.pi / 2)), 1.,
         np.array([0, 0, 0, 0, 0, 0, 1, 0,
                      0, 0, 1, 0, 1, 0, 0, 0]) / np.sqrt(3),
         np.array([0, 0, 0, 1j, 0, -1j / 2., 1 / np.sqrt(2), 0,
                      0, 1j / np.sqrt(2), np.sqrt(3) / 2, 0, 0, 0, 0, 0]
                     ) / np.sqrt(3),
         5e-6),
        ]
@pytest.mark.parametrize(
    'gate, exponent, initial_state, correct_state, atol',
    quartic_fermionic_simulation_simulator_test_cases)
def test_quartic_fermionic_simulation_on_simulator(
        gate, exponent, initial_state, correct_state, atol):

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(gate(a, b, c, d)**exponent)
    result = circuit.apply_unitary_effect_to_state(initial_state)
    cirq.testing.assert_allclose_up_to_global_phase(
        result, correct_state, atol=atol)


def test_quartic_fermionic_simulation_init_with_multiple_args_fails():
    with pytest.raises(ValueError):
        _ = ofc.QuarticFermionicSimulationGate(
                (1,1,1), exponent=1.0, duration=np.pi/2)


def test_quartic_fermionic_simulation_eq():
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(
            ofc.QuarticFermionicSimulationGate((1.2, 0.4, -0.4), exponent=0.5),
            ofc.QuarticFermionicSimulationGate((0.3, 0.1, -0.1), exponent=2),
            ofc.QuarticFermionicSimulationGate((-0.6, -0.2, 0.2), exponent=-1),
            ofc.QuarticFermionicSimulationGate((0.6, 0.2, 2 * np.pi - 0.2)),
            ofc.QuarticFermionicSimulationGate(
                (1.2, 0.4, -0.4), rads=0.5 * np.pi),
            ofc.QuarticFermionicSimulationGate((1.2, 0.4, -0.4), degs=90),
            ofc.QuarticFermionicSimulationGate(
                (1.2, 0.4, -0.4), duration=0.5 * np.pi / 2)
            )

    eq.add_equality_group(
            ofc.QuarticFermionicSimulationGate((-0.6, 0.0, 0.3), exponent=0.5),
            ofc.QuarticFermionicSimulationGate((-0.6, 0.0, 0.3),
                                             rads=0.5 * np.pi),
            ofc.QuarticFermionicSimulationGate((-0.6, 0.0, 0.3), degs=90))

    eq.make_equality_group(
            lambda: ofc.QuarticFermionicSimulationGate(
                (0.1, -0.3, 0.0), exponent=0.0))
    eq.make_equality_group(
            lambda: ofc.QuarticFermionicSimulationGate(
                (1., -1., 0.5), exponent=0.75))


def test_quartic_fermionic_simulation_gate_text_diagram():
    gate = ofc.QuarticFermionicSimulationGate((1,1,1))
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([gate(*qubits[:4]), gate(*qubits[-4:])])

    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
0: ───⇊⇈────────
      │
1: ───⇊⇈────────
      │
2: ───⇊⇈───⇊⇈───
      │    │
3: ───⇊⇈───⇊⇈───
           │
4: ────────⇊⇈───
           │
5: ────────⇊⇈───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    actual_text_diagram = circuit.to_text_diagram(use_unicode_characters=False)
    expected_text_diagram = """
0: ---a*a*aa------------
      |
1: ---a*a*aa------------
      |
2: ---a*a*aa---a*a*aa---
      |        |
3: ---a*a*aa---a*a*aa---
               |
4: ------------a*a*aa---
               |
5: ------------a*a*aa---
    """.strip()
    assert actual_text_diagram == expected_text_diagram


test_weights = [1.0, 0.5, 0.25, 0.1, 0.0, -0.5]
@pytest.mark.parametrize('weights', itertools.chain(
        itertools.product(test_weights, repeat=3),
        np.random.rand(10, 3)
        ))
def test_quartic_fermionic_simulation_decompose(weights):
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        ofc.QuarticFermionicSimulationGate(weights))


@pytest.mark.parametrize('weights,exponent', [
    (np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
        np.random.uniform(-5, 5)) for _ in range(5)
])
def test_quartic_fermionic_simulation_unitary(
        weights, exponent):
    generator = np.zeros((1 << 4,) * 2, dtype=np.complex128)

    # w0 |1001><0110| + h.c.
    generator[9, 6] = weights[0]
    generator[6, 9] = weights[0].conjugate()
    # w1 |1010><0101| + h.c.
    generator[10, 5] = weights[1]
    generator[5, 10] = weights[1].conjugate()
    # w2 |1100><0011| + h.c.
    generator[12, 3] = weights[2]
    generator[3, 12] = weights[2].conjugate()
    expected_unitary = la.expm(-1j * exponent * generator)

    gate  = ofc.QuarticFermionicSimulationGate(weights, exponent=exponent)
    actual_unitary = cirq.unitary(gate)

    assert np.allclose(expected_unitary, actual_unitary)

    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


@pytest.mark.parametrize('weights,exponent', [
    (np.random.uniform(-5, 5, 3) + 1j * np.random.uniform(-5, 5, 3),
        np.random.uniform(-5, 5)) for _ in range(5)
])
def test_quartic_fermionic_simulation_apply_unitary(weights, exponent):
    gate = ofc.QuarticFermionicSimulationGate(weights, exponent=exponent)
    cirq.testing.assert_has_consistent_apply_unitary(gate, atol=5e-6)
