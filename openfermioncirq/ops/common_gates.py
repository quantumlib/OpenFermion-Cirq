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

from typing import Optional, Union

import numpy

import cirq


class FermionicSwapGate(cirq.InterchangeableQubitsGate,
                        cirq.KnownMatrixGate,
                        cirq.ReversibleEffect,
                        cirq.TextDiagrammableGate,
                        cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def matrix(self):
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])

    def inverse(self):
        return self

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '×ᶠ', '×ᶠ'

    def __repr__(self):
        return 'FSWAP'


class XXYYGate(cirq.EigenGate,
               cirq.CompositeGate,
               cirq.InterchangeableQubitsGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """XX + YY interaction."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

        There are two ways to instantiate this gate.
        
        The first is to provide an angle in units of half-turns. In this case,
        the gate implements the unitary exp(-i pi half_turns (XX + YY) / 4).
        
        The second way is to provide a duration of time. In this case, the gate
        implements the unitary exp(-i duration (XX + YY) / 2 ), which
        corresponds to evolving under the Hamiltonian (XX + YY) / 2 for that
        duration of time.

        At most one argument can be specified. If both `half_turns` and
        `duration` are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            half_turns: The exponent angle, in half-turns.
            duration: The exponent duration.
        """
        if len([1 for e in [half_turns, duration] if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        elif half_turns is not None:
            exponent = half_turns
        else:
            exponent = 1.0

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
            (0.5, numpy.array([[0, 0, 0, 0],
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'XXYY', 'XXYY'

    def text_diagram_exponent(self):
        return self.half_turns

    def __repr__(self):
        if self.half_turns == 1:
            return 'XXYY'
        return 'XXYY**{!r}'.format(self.half_turns)


class YXXYGate(cirq.EigenGate,
               cirq.CompositeGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """YX - XY interaction."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

        There are two ways to instantiate this gate.
        
        The first is to provide an angle in units of half-turns. In this case,
        the gate implements the unitary exp(-i pi half_turns (YX - XY) / 4).
        
        The second way is to provide a duration of time. In this case, the gate
        implements the unitary exp(-i duration (YX - XY) / 2 ), which
        corresponds to evolving under the Hamiltonian (YX - XY) / 2 for that
        duration of time.

        At most one argument can be specified. If both `half_turns` and
        `duration` are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            half_turns: The exponent angle, in half-turns.
            duration: The exponent duration.
        """
        if len([1 for e in [half_turns, duration] if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        elif half_turns is not None:
            exponent = half_turns
        else:
            exponent = 1.0

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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'YXXY', '#2'

    def text_diagram_exponent(self):
        return self.half_turns

    def __repr__(self):
        if self.half_turns == 1:
            return 'YXXY'
        return 'YXXY**{!r}'.format(self.half_turns)


class ZZGate(cirq.EigenGate,
             cirq.TwoQubitGate,
             cirq.TextDiagrammableGate,
             cirq.InterchangeableQubitsGate):
    """ZZ interaction."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

        There are two ways to instantiate this gate.
        
        The first is to provide an angle in units of half-turns. In this case,
        the gate implements the unitary exp(-i pi half_turns ZZ / 2).
        
        The second way is to provide a duration of time. In this case, the gate
        implements the unitary exp(-i duration ZZ), which corresponds to
        evolving under the Hamiltonian ZZ for that duration of time.

        At most one argument can be specified. If both `half_turns` and
        `duration` are specified, the result is considered ambiguous and an
        error is thrown. If no argument is given, the default value of one
        half-turn is used.

        Args:
            half_turns: The exponent angle, in half-turns.
            duration: The exponent duration.
        """
        if len([1 for e in [half_turns, duration] if e is not None]) > 1:
            raise ValueError('Redundant exponent specification. '
                             'Use ONE of half_turns or duration.')

        if duration is not None:
            exponent = 2 * duration / numpy.pi
        elif half_turns is not None:
            exponent = half_turns
        else:
            exponent = 1.0

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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'Z', 'Z'

    def text_diagram_exponent(self):
        return self.half_turns

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'ZZ'
        return 'ZZ**{!r}'.format(self.half_turns)


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
YXXY = YXXYGate()
ZZ = ZZGate()

ISWAP = XXYY
