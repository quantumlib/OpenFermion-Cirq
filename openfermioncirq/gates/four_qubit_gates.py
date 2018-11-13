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

"""Gates that target four qubits."""


from typing import Optional, Union, Tuple, Sequence

import numpy as np

import cirq
from cirq.type_workarounds import NotImplementedType


def state_swap_eigen_component(x: str, y: str, sign: int = 1):
    """The +/- eigen-component of the operation that swaps states x and y.

    For example, state_swap_eigen_component('01', '10', ±1) returns
        ┌             ┐
        │0 0    0    0│
        │0 0.5  ±0.5 0│
        │0 ±0.5 0.5  0│
        │0 0    0    0│
        └             ┘

    Args:
        x: The first state to swap, as a bitstring.
        y: The second state to swap, as a bitstring.
        sign: The sign of the off-diagonal elements (indicated by +/-1).

    Returns: The eigen-component.

    Raises:
        ValueError:
            * x and y have different lengths
            * x or y contains a character other than '0' and '1'
            * x and y are the same
            * sign is not -1 or 1
        TypeError: x or y is not a string
    """
    if not (isinstance(x, str) and isinstance(y, str)):
        raise TypeError('not (isinstance(x, str) and isinstance(y, str))')
    if len(x) != len(y):
        raise ValueError('len(x) != len(y)')
    if set(x).union(y).difference('01'):
        raise ValueError('Arguments must be 0-1 strings.')
    if x == y:
        raise ValueError('x == y')
    if sign not in (-1, 1):
        raise ValueError('sign not in (-1, 1)')

    dim = 2 ** len(x)
    i, j = int(x, 2), int(y, 2)

    component = np.zeros((dim, dim))
    component[i, i] = component[j, j] = 0.5
    component[i, j] = component[j, i] = sign * 0.5
    return component


class DoubleExcitationGate(cirq.EigenGate):
    """Evolve under -|0011><1100| + h.c. for some time."""

    def __init__(self, *,  # Forces keyword args.
                 exponent: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initialize the gate.

        At most one of exponent, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            exponent: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        if len([1 for e in [exponent, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of exponent, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / np.pi
        else:
            exponent = cirq.chosen_angle_to_half_turns(
                half_turns=exponent,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

    def _eigen_components(self):
        minus_one_component = np.zeros((16, 16))
        minus_one_component[3, 3] = minus_one_component[12, 12] = 0.5
        minus_one_component[3, 12] = minus_one_component[12, 3] = -0.5

        plus_one_component = np.zeros((16, 16))
        plus_one_component[3, 3] = plus_one_component[12, 12] = 0.5
        plus_one_component[3, 12] = plus_one_component[12, 3] = 0.5

        return [(0, np.diag([1, 1, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 1, 1])),
                (-1, minus_one_component),
                (1, plus_one_component)]

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if cirq.is_parameterized(self):
            return NotImplemented
        inner_matrix = cirq.unitary(cirq.Rx(-2*np.pi*self.exponent))
        a = cirq.slice_for_qubits_equal_to(axes, 0b0011)
        b = cirq.slice_for_qubits_equal_to(axes, 0b1100)
        return cirq.apply_matrix_to_slices(target_tensor,
                                           inner_matrix,
                                           slices=[a, b],
                                           out=available_buffer)

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'DoubleExcitationGate':
        return DoubleExcitationGate(exponent=exponent)

    def _decompose_(self, qubits):
        p, q, r, s = qubits

        rq_phase_block = [cirq.Z(q) ** 0.125,
                          cirq.CNOT(r, q),
                          cirq.Z(q) ** -0.125]

        srq_parity_transform = [cirq.CNOT(s, r),
                                cirq.CNOT(r, q),
                                cirq.CNOT(s, r)]

        phase_parity_block = [[rq_phase_block,
                              srq_parity_transform,
                              rq_phase_block]]

        yield cirq.CNOT(r, s)
        yield cirq.CNOT(q, p)
        yield cirq.CNOT(q, r)
        yield cirq.X(q) ** -self.exponent
        yield phase_parity_block

        yield cirq.CNOT(p, q)
        yield cirq.X(q)
        yield phase_parity_block
        yield cirq.X(q) ** self.exponent
        yield phase_parity_block
        yield cirq.CNOT(p, q)
        yield cirq.X(q)

        yield phase_parity_block
        yield cirq.CNOT(q, p)
        yield cirq.CNOT(q, r)
        yield cirq.CNOT(r, s)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇅', '⇅', '⇵', '⇵')
        else:
            wire_symbols = ('/\\ \/',
                            '/\\ \/',
                            '\/ /\\',
                            '\/ /\\')
        return cirq.CircuitDiagramInfo(
            wire_symbols=wire_symbols,
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'DoubleExcitation'
        return 'DoubleExcitation**{!r}'.format(self.exponent)


DoubleExcitation = DoubleExcitationGate()


class CombinedDoubleExcitationGate(cirq.EigenGate):
    """Rotates Hamming-weight 2 states into their bitwise complements.

    For weights (t0, t1, t2), is equivalent to
        exp(0.5 pi i (t0 (|1001><0110| + |0110><1001|) +
                      t1 (|0101><1010| + |1010><0101|) +
                      t2 (|0011><1100| + |1100><0011|)))
    """

    def __init__(self,
                 weights: Tuple[float, float, float]=(1, 1, 1),
                 absorb_exponent: bool=True,
                 *,  # Forces keyword args.
                 exponent: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None
                 ) -> None:
        """Initialize the gate.

        At most one of exponent, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            weights: The weights of the terms in the Hamiltonian.
            absorb_exponent: Whether to absorb the given exponent into the
                weights. If true, the exponent of the returned gate is 1.
            exponent: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        self.weights = weights

        if len([1 for e in [exponent, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of exponent, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / np.pi
        else:
            exponent = cirq.chosen_angle_to_half_turns(
                half_turns=exponent,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

        if absorb_exponent:
            self.absorb_exponent_into_weights()

    def _eigen_components(self):
        # projector onto subspace spanned by basis states with
        # Hamming weight != 2
        zero_component = np.diag([int(bin(i).count('1') != 2)
                                  for i in range(16)])

        state_pairs = (('1001', '0110'),
                       ('0101', '1010'),
                       ('0011', '1100'))

        plus_minus_components = tuple(
            (weight * sign / 2,
             state_swap_eigen_component(state_pair[0], state_pair[1], sign))
            for weight, state_pair in zip(self.weights, state_pairs)
            for sign in (-1, 1))

        return ((0, zero_component),) + plus_minus_components

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'CombinedDoubleExcitationGate':
        gate = CombinedDoubleExcitationGate(self.weights)
        gate._exponent = exponent
        return gate

    def _decompose_(self, qubits):
        a, b, c, d = qubits

        weights_to_exponents = (self._exponent / 4.) * np.array([
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1]
        ])
        exponents = weights_to_exponents.dot(self.weights)

        basis_change = list(cirq.flatten_op_tree([
            cirq.CNOT(b, a),
            cirq.CNOT(c, b),
            cirq.CNOT(d, c),
            cirq.CNOT(c, b),
            cirq.CNOT(b, a),
            cirq.CNOT(a, b),
            cirq.CNOT(b, c),
            cirq.CNOT(a, b),
            [cirq.X(c), cirq.X(d)],
            [cirq.CNOT(c, d), cirq.CNOT(d, c)],
            [cirq.X(c), cirq.X(d)],
            ]))

        controlled_Zs = list(cirq.flatten_op_tree([
            cirq.CZPowGate(exponent=exponents[0])(b, c),
            cirq.CNOT(a, b),
            cirq.CZPowGate(exponent=exponents[1])(b, c),
            cirq.CNOT(b, a),
            cirq.CNOT(a, b),
            cirq.CZPowGate(exponent=exponents[2])(b, c)
            ]))

        controlled_swaps = [
            [cirq.CNOT(c, d), cirq.H(c)],
            cirq.CNOT(d, c),
            controlled_Zs,
            cirq.CNOT(d, c),
            [cirq.inverse(op) for op in reversed(controlled_Zs)],
            [cirq.H(c), cirq.CNOT(c, d)],
            ]

        yield basis_change
        yield controlled_swaps
        yield basis_change[::-1]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇊⇈',) * 4
        else:
            wire_symbols = ('a*a*aa',) * 4
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols,
                                       exponent=self._diagram_exponent(args))

    def absorb_exponent_into_weights(self):
        self.weights = tuple((w * self._exponent) % 4 for w in self.weights)
        self._exponent = 1

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if cirq.is_parameterized(self):
            return NotImplemented
        am = cirq.unitary(cirq.Rx(-np.pi * self.exponent * self.weights[0]))
        bm = cirq.unitary(cirq.Rx(-np.pi * self.exponent * self.weights[1]))
        cm = cirq.unitary(cirq.Rx(-np.pi * self.exponent * self.weights[2]))

        a1 = cirq.slice_for_qubits_equal_to(axes, 0b1001)
        b1 = cirq.slice_for_qubits_equal_to(axes, 0b0101)
        c1 = cirq.slice_for_qubits_equal_to(axes, 0b0011)

        a2 = cirq.slice_for_qubits_equal_to(axes, 0b0110)
        b2 = cirq.slice_for_qubits_equal_to(axes, 0b1010)
        c2 = cirq.slice_for_qubits_equal_to(axes, 0b1100)

        cirq.apply_matrix_to_slices(target_tensor,
                                    am,
                                    slices=[a1, a2],
                                    out=available_buffer)
        cirq.apply_matrix_to_slices(available_buffer,
                                    bm,
                                    slices=[b1, b2],
                                    out=target_tensor)
        return cirq.apply_matrix_to_slices(target_tensor,
                                           cm,
                                           slices=[c1, c2],
                                           out=available_buffer)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(np.isclose((w * self._exponent) % 4,
                              (ww * other._exponent) % 4)
                   for w, ww in zip(self.weights, other.weights))

    def __repr__(self):
        weights = tuple(w * self._exponent for w in self.weights)
        return 'CombinedDoubleExcitation' + str(weights)
