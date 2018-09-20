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


from typing import Optional, Union, Tuple

import numpy

import cirq


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
        x, y: The states to swap, as bitstrings.
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

    component = numpy.zeros((dim, dim))
    component[i, i] = component[j, j] = 0.5
    component[i, j] = component[j, i] = sign * 0.5
    return component


class DoubleExcitationGate(cirq.EigenGate,
                           cirq.CompositeGate,
                           cirq.TextDiagrammable):
    """Evolve under -|0011><1100| + h.c. for some time."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initialize the gate.

        At most one of half_turns, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            half_turns: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        if len([1 for e in [half_turns, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        else:
            exponent = cirq.value.chosen_angle_to_half_turns(
                half_turns=half_turns,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        minus_one_component = numpy.zeros((16, 16))
        minus_one_component[3, 3] = minus_one_component[12, 12] = 0.5
        minus_one_component[3, 12] = minus_one_component[12, 3] = -0.5

        plus_one_component = numpy.zeros((16, 16))
        plus_one_component[3, 3] = plus_one_component[12, 12] = 0.5
        plus_one_component[3, 12] = plus_one_component[12, 3] = 0.5

        return [(0, numpy.diag([1, 1, 1, 0, 1, 1, 1, 1,
                                1, 1, 1, 1, 0, 1, 1, 1])),
                (-1, minus_one_component),
                (1, plus_one_component)]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'DoubleExcitationGate':
        return DoubleExcitationGate(half_turns=exponent)

    def default_decompose(self, qubits):
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
        yield cirq.X(q) ** -self.half_turns
        yield phase_parity_block

        yield cirq.CNOT(p, q)
        yield cirq.X(q)
        yield phase_parity_block
        yield cirq.X(q) ** self.half_turns
        yield phase_parity_block
        yield cirq.CNOT(p, q)
        yield cirq.X(q)

        yield phase_parity_block
        yield cirq.CNOT(q, p)
        yield cirq.CNOT(q, r)
        yield cirq.CNOT(r, s)

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇅', '⇅', '⇵', '⇵')
        else:
            wire_symbols = ('/\\ \/',
                            '/\\ \/',
                            '\/ /\\',
                            '\/ /\\')
        return cirq.TextDiagramInfo(wire_symbols=wire_symbols,
                                    exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'DoubleExcitation'
        return 'DoubleExcitation**{!r}'.format(self.half_turns)


DoubleExcitation = DoubleExcitationGate()


class CombinedDoubleExcitationGate(cirq.EigenGate,
                           cirq.CompositeGate,
                           cirq.TextDiagrammable):
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
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None
                 ) -> None:
        """Initialize the gate.

        At most one of half_turns, rads, degs, or duration may be specified.
        If more are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            weights: The weights of the terms in the Hamiltonian.
            absorb_exponent: Whether to absorb the given exponent into the
                weights. If true, the exponent of the returned gate is 1.
            half_turns: The exponent angle, in half-turns.
            rads: The exponent angle, in radians.
            degs: The exponent angle, in degrees.
            duration: The exponent as a duration of time.
        """

        self.weights = weights

        if len([1 for e in [half_turns, rads, degs, duration]
                if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns, rads, degs, or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        else:
            exponent = cirq.value.chosen_angle_to_half_turns(
                half_turns=half_turns,
                rads=rads,
                degs=degs)

        super().__init__(exponent=exponent)

        if absorb_exponent:
            self.absorb_exponent_into_weights()

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        # projector onto subspace spanned by basis states with
        # Hamming weight != 2
        zero_component = numpy.diag([int(bin(i).count('1') != 2)
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

    def _canonical_exponent_period(self) -> Optional[float]:
        return None

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'CombinedDoubleExcitationGate':
        gate = CombinedDoubleExcitationGate(self.weights)
        gate._exponent = exponent
        return gate

    def default_decompose(self, qubits):
        a, b, c, d = qubits

        weights_to_exponents = (self._exponent / 4.) * numpy.array([
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
            cirq.Rot11Gate(half_turns=exponents[0])(b, c),
            cirq.CNOT(a, b),
            cirq.Rot11Gate(half_turns=exponents[1])(b, c),
            cirq.CNOT(b, a),
            cirq.CNOT(a, b),
            cirq.Rot11Gate(half_turns=exponents[2])(b, c)
            ]))

        controlled_swaps = [
            [cirq.CNOT(c, d), cirq.H(c)],
            cirq.CNOT(d, c),
            controlled_Zs,
            cirq.CNOT(d, c),
            [op.inverse() for op in reversed(controlled_Zs)],
            [cirq.H(c), cirq.CNOT(c, d)],
            ]

        yield basis_change
        yield controlled_swaps
        yield basis_change[::-1]

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇊⇈',) * 4
        else:
            wire_symbols = ('a*a*aa',) * 4
        return cirq.TextDiagramInfo(wire_symbols=wire_symbols,
                                    exponent=self.half_turns)

    def absorb_exponent_into_weights(self):
        self.weights = tuple((w * self._exponent) % 4 for w in self.weights)
        self._exponent = 1


    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(numpy.isclose((w * self._exponent) % 4,
                                 (ww * other._exponent) % 4)
                   for w, ww in zip(self.weights, other.weights))

    def __repr__(self):
        weights = tuple(w * self._exponent for w in self.weights)
        return 'CombinedDoubleExcitation' + str(weights)
