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


class FermionicSwapGate(cirq.TextDiagrammableGate,
                        cirq.InterchangeableQubitsGate,
                        cirq.KnownMatrixGate,
                        cirq.SelfInverseGate,
                        cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def matrix(self):
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '×ᶠ', '×ᶠ'

    def __repr__(self):
        return 'FSWAP'


class XXYYGate(cirq.CompositeGate,
               cirq.EigenGate,
               cirq.InterchangeableQubitsGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """XX + YY interaction.

    This gate implements the unitary exp(-i pi quarter_turns (XX + YY) / 4)
    """

    def __init__(self, *positional_args,
                 quarter_turns: float=1.0) -> None:
        assert not positional_args
        super().__init__(exponent=quarter_turns)

    @property
    def quarter_turns(self) -> Union[cirq.Symbol, float]:
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
        return XXYYGate(quarter_turns=exponent)

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** 0.5
        yield YXXY(a, b) ** self.quarter_turns
        yield cirq.Z(a) ** -0.5

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'XXYY', 'XXYY'

    def text_diagram_exponent(self):
        return self.quarter_turns

    def __repr__(self):
        if self.quarter_turns == 1:
            return 'XXYY'
        return 'XXYY**{!r}'.format(self.quarter_turns)


class YXXYGate(cirq.CompositeGate,
               cirq.EigenGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """YX - XY interaction.

    This gate implements the unitary exp(-i pi quarter_turns (YX - XY) / 4)
    """

    def __init__(self, *positional_args,
                 quarter_turns: float=1.0) -> None:
        assert not positional_args
        super().__init__(exponent=quarter_turns)

    @property
    def quarter_turns(self) -> Union[cirq.Symbol, float]:
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
        return YXXYGate(quarter_turns=exponent)

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** -0.5
        yield XXYY(a, b) ** self.quarter_turns
        yield cirq.Z(a) ** 0.5

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'YXXY', '#2'

    def text_diagram_exponent(self):
        return self.quarter_turns

    def __repr__(self):
        if self.quarter_turns == 1:
            return 'YXXY'
        return 'YXXY**{!r}'.format(self.quarter_turns)


class Rot111Gate(cirq.CompositeGate,
                 cirq.EigenGate,
                 cirq.InterchangeableQubitsGate,
                 cirq.TextDiagrammableGate):
    """Phases the |111> state of three qubits by a fixed amount."""

    def __init__(self,
                 *positional_args,
                 half_turns: Union[cirq.Symbol, float] = 1.0) -> None:
        assert not positional_args
        super().__init__(exponent=half_turns)

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (0, numpy.diag([1, 1, 1, 1, 1, 1, 1, 0])),
            (1, numpy.diag([0, 0, 0, 0, 0, 0, 0, 1])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]) -> 'Rot111Gate':
        return Rot111Gate(half_turns=exponent)

    def default_decompose(self, qubits):
        a, b, c = qubits
        yield cirq.CZ(b, c)**(0.5 * self.half_turns)
        yield cirq.CNOT(a, b)
        yield cirq.CZ(b, c)**(-0.5 * self.half_turns)
        yield cirq.CNOT(a, b)
        yield cirq.CZ(a, c)**(0.5 * self.half_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', '@', 'Z'

    def text_diagram_exponent(self):
        return self.half_turns

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CCZ'
        return 'CCZ**{!r}'.format(self.half_turns)


class ControlledXXYYGate(cirq.CompositeGate,
                         cirq.EigenGate,
                         cirq.TextDiagrammableGate):
    """Controlled XX + YY interaction."""

    def __init__(self, *positional_args,
                 quarter_turns: float=1.0) -> None:
        assert not positional_args
        super().__init__(exponent=quarter_turns)

    @property
    def quarter_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (0, numpy.diag([1, 1, 1, 1, 1, 0, 0, 1])),
            (-0.5, numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0.5, 0.5, 0],
                                [0, 0, 0, 0, 0, 0.5, 0.5, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]])),
            (0.5, numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0.5, -0.5, 0],
                               [0, 0, 0, 0, 0, -0.5, 0.5, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'ControlledXXYYGate':
        return ControlledXXYYGate(quarter_turns=exponent)

    def default_decompose(self, qubits):
        control, a, b = qubits
        yield cirq.Z(a)
        yield cirq.Y(a)**-0.5, cirq.Y(b)**-0.5
        yield CCZ(control, a, b)**self.quarter_turns
        yield cirq.CZ(control, a)**(-0.5 * self.quarter_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.quarter_turns)
        yield cirq.Y(a)**0.5, cirq.Y(b)**0.5
        yield cirq.X(a)**0.5, cirq.X(b)**0.5
        yield CCZ(control, a, b)**self.quarter_turns
        yield cirq.CZ(control, a)**(-0.5 * self.quarter_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.quarter_turns)
        yield cirq.X(a)**-0.5, cirq.X(b)**-0.5
        yield cirq.Z(a)
        yield cirq.Z(control)**(0.5 * self.quarter_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'XXYY', 'XXYY'

    def text_diagram_exponent(self):
        return self.quarter_turns

    def __repr__(self):
        if self.quarter_turns == 1:
            return 'CXXYY'
        return 'CXXYY**{!r}'.format(self.quarter_turns)


class ControlledYXXYGate(cirq.CompositeGate,
                         cirq.EigenGate,
                         cirq.TextDiagrammableGate):
    """Controlled YX - XY interaction."""

    def __init__(self, *positional_args,
                 quarter_turns: float=1.0) -> None:
        assert not positional_args
        super().__init__(exponent=quarter_turns)

    @property
    def quarter_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (0, numpy.diag([1, 1, 1, 1, 1, 0, 0, 1])),
            (-0.5, numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0.5, -0.5j, 0],
                                [0, 0, 0, 0, 0, 0.5j, 0.5, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]])),
            (0.5, numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0.5, 0.5j, 0],
                               [0, 0, 0, 0, 0, -0.5j, 0.5, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]]))
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'ControlledYXXYGate':
        return ControlledYXXYGate(quarter_turns=exponent)

    def default_decompose(self, qubits):
        control, a, b = qubits
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=5/8).on(a)
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=7/8).on(b)
        yield cirq.Y(a)**-0.5, cirq.Y(b)**-0.5
        yield CCZ(control, a, b)**self.quarter_turns
        yield cirq.CZ(control, a)**(-0.5 * self.quarter_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.quarter_turns)
        yield cirq.Y(a)**0.5, cirq.Y(b)**0.5
        yield cirq.X(a)**0.5, cirq.X(b)**0.5
        yield CCZ(control, a, b)**self.quarter_turns
        yield cirq.CZ(control, a)**(-0.5 * self.quarter_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.quarter_turns)
        yield cirq.X(a)**-0.5, cirq.X(b)**-0.5
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=5/8).on(a)
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=7/8).on(b)
        yield cirq.Z(control)**(0.5 * self.quarter_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'YXXY', '#2'

    def text_diagram_exponent(self):
        return self.quarter_turns

    def __repr__(self):
        if self.quarter_turns == 1:
            return 'CYXXY'
        return 'CYXXY**{!r}'.format(self.quarter_turns)


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
YXXY = YXXYGate()
CCZ = Rot111Gate()
CXXYY = ControlledXXYYGate()
CYXXY = ControlledYXXYGate()
