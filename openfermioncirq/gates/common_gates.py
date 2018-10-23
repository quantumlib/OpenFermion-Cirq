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

"""Gates that are commonly used for quantum simulation of fermions."""

from typing import Optional, Union, Tuple

import numpy

import cirq


class FermionicSwapGate(cirq.InterchangeableQubitsGate, cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def _unitary_(self) -> numpy.ndarray:
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])

    def __pow__(self, power) -> 'FermionicSwapGate':
        if power in [1, -1]:
            return self
        # coverage: ignore
        return NotImplemented

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> Tuple[str, str]:
        if args.use_unicode_characters:
            return '×ᶠ', '×ᶠ'
        return 'fswap', 'fswap'

    def __repr__(self):
        return 'FSWAP'


class XXYYGate(cirq.EigenGate,
               cirq.CompositeGate,
               cirq.InterchangeableQubitsGate,
               cirq.TwoQubitGate):
    """XX + YY interaction.

    There are two ways to instantiate this gate.

    The first is to provide an angle in units of either half-turns,
    radians, or degrees. In this case, the gate's matrix is defined
    as follows::

        XXYY**h ≡ exp(-i π h (X⊗X + Y⊗Y) / 4)
                ≡ exp(-i rads (X⊗X + Y⊗Y) / 4)
                ≡ exp(-i π (degs / 180) (X⊗X + Y⊗Y) / 4)
                ≡ [1 0             0             0]
                  [0 cos(π·h/2)    -i·sin(π·h/2) 0]
                  [0 -i·sin(π·h/2) cos(π·h/2)    0]
                  [0 0             0             1]

    where h is the angle in half-turns.

    The second way is to provide a duration of time. In this case, the gate
    implements the unitary::

        exp(-i t (X⊗X + Y⊗Y) / 2) ≡ [1 0         0         0]
                                    [0 cos(t)    -i·sin(t) 0]
                                    [0 -i·sin(t) cos(t)    0]
                                    [0 0         0         1]

    where t is the duration. This corresponds to evolving under the
    Hamiltonian (X⊗X + Y⊗Y) / 2 for that duration of time.
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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
        return [
            (0, numpy.diag([1, 0, 0, 1])),
            (-0.5, numpy.array([[0, 0, 0, 0],
                                [0, 0.5, 0.5, 0],
                                [0, 0.5, 0.5, 0],
                                [0, 0, 0, 0]])),
            (+0.5, numpy.array([[0, 0, 0, 0],
                                [0, 0.5, -0.5, 0],
                                [0, -0.5, 0.5, 0],
                                [0, 0, 0, 0]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self, exponent: Union[cirq.Symbol, float]) -> 'XXYYGate':
        return XXYYGate(half_turns=exponent)

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** 0.5
        yield YXXY(a, b) ** self.half_turns
        yield cirq.Z(a) ** -0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('XXYY', 'XXYY'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'XXYY'
        return 'XXYY**{!r}'.format(self.half_turns)


class YXXYGate(cirq.EigenGate,
               cirq.CompositeGate,
               cirq.TwoQubitGate):
    """YX - XY interaction.

    There are two ways to instantiate this gate.

    The first is to provide an angle in units of either half-turns,
    radians, or degrees. In this case, the gate's matrix is defined
    as follows::

        YXXY**h ≡ exp(-i π h (Y⊗X - X⊗Y) / 4)
                ≡ exp(-i rads (Y⊗X - X⊗Y) / 4)
                ≡ exp(-i π (degs / 180) (Y⊗X - X⊗Y) / 4)
                ≡ [1 0          0           0]
                  [0 cos(π·h/2) -sin(π·h/2) 0]
                  [0 sin(π·h/2) cos(π·h/2)  0]
                  [0 0          0           1]

    where h is the angle in half-turns.

    The second way is to provide a duration of time. In this case, the gate
    implements the unitary::

        exp(-i t (Y⊗X - X⊗Y) / 2) ≡ [1 0      0       0]
                                    [0 cos(t) -sin(t) 0]
                                    [0 sin(t) cos(t)  0]
                                    [0 0      0       1]

    where t is the duration. This corresponds to evolving under the
    Hamiltonian (Y⊗X - X⊗Y) / 2 for that duration of time.
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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
        return [
            (0, numpy.diag([1, 0, 0, 1])),
            (-0.5, numpy.array([[0, 0, 0, 0],
                                [0, 0.5, -0.5j, 0],
                                [0, 0.5j, 0.5, 0],
                                [0, 0, 0, 0]])),
            (0.5, numpy.array([[0, 0, 0, 0],
                               [0, 0.5, 0.5j, 0],
                               [0, -0.5j, 0.5, 0],
                               [0, 0, 0, 0]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self, exponent: Union[cirq.Symbol, float]) -> 'YXXYGate':
        return YXXYGate(half_turns=exponent)

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** -0.5
        yield XXYY(a, b) ** self.half_turns
        yield cirq.Z(a) ** 0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('YXXY', '#2'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'YXXY'
        return 'YXXY**{!r}'.format(self.half_turns)


class ZZGate(cirq.EigenGate,
             cirq.TwoQubitGate,
             cirq.InterchangeableQubitsGate):
    """ZZ interaction.

    There are two ways to instantiate this gate.

    The first is to provide an angle in units of either half-turns,
    radians, or degrees. In this case, the gate's matrix is defined
    as follows::

        ZZ**h ≡ exp(-i π h (Z⊗Z) / 2)
              ≡ exp(-i rads (Z⊗Z) / 2)
              ≡ exp(-i π (degs / 180) (Z⊗Z) / 2)
              ≡ [exp(-i·π·h/2) 0             0                         0]
                [0             exp(+i·π·h/2) 0                         0]
                [0             0             exp(+i·π·h/2)             0]
                [0             0             0             exp(-i·π·h/2)]

    where h is the angle in half-turns. At a value of one half-turn, this
    gate is equivalent to Z⊗Z = diag(1, -1, -1, 1) up to a global phase.
    More generally, ZZ**h is equivalent to diag(1, (-1)**h, (-1)**h, 1)
    up to a global phase.

    The second way to instantiate this gate is to provide a duration
    of time. In this case, the gate implements the unitary::

        exp(-i t Z⊗Z) ≡ [exp(-it) 0          0               0]
                        [0          exp(+it) 0               0]
                        [0          0        exp(+it)        0]
                        [0          0        0        exp(-it)]

    where t is the duration. This corresponds to evolving under the
    Hamiltonian Z⊗Z for that duration of time.
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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
        return [
            (-0.5, numpy.diag([1, 0, 0, 1])),
            (0.5, numpy.diag([0, 1, 1, 0])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]) -> 'ZZGate':
        return ZZGate(half_turns=exponent)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('Z', 'Z'),
            exponent=self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'ZZ'
        return 'ZZ**{!r}'.format(self.half_turns)


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
YXXY = YXXYGate()
ZZ = ZZGate()
