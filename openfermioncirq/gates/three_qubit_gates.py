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

"""Common gates that target three qubits."""

from typing import Optional, Union

import numpy

import cirq


class Rot111Gate(cirq.EigenGate,
                 cirq.CompositeGate,
                 cirq.InterchangeableQubitsGate,
                 cirq.TextDiagrammable):
    """Phases the |111> state of three qubits by a fixed amount."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: Relative phasing of CCZ's eigenstates, in half_turns.
            rads: Relative phasing of CCZ's eigenstates, in radians.
            degs: Relative phasing of CCZ's eigenstates, in degrees.
        """
        super().__init__(exponent=cirq.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

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

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        return cirq.TextDiagramInfo(
            wire_symbols=('@', '@', '@'),
            exponent=self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CCZ'
        return 'CCZ**{!r}'.format(self.half_turns)


class ControlledXXYYGate(cirq.EigenGate,
                         cirq.CompositeGate,
                         cirq.TextDiagrammable):
    """Controlled XX + YY interaction."""
    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:

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
        return ControlledXXYYGate(half_turns=exponent)

    def default_decompose(self, qubits):
        control, a, b = qubits
        yield cirq.Z(a)
        yield cirq.Y(a)**-0.5, cirq.Y(b)**-0.5
        yield CCZ(control, a, b)**self.half_turns
        yield cirq.CZ(control, a)**(-0.5 * self.half_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.half_turns)
        yield cirq.Y(a)**0.5, cirq.Y(b)**0.5
        yield cirq.X(a)**0.5, cirq.X(b)**0.5
        yield CCZ(control, a, b)**self.half_turns
        yield cirq.CZ(control, a)**(-0.5 * self.half_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.half_turns)
        yield cirq.X(a)**-0.5, cirq.X(b)**-0.5
        yield cirq.Z(a)
        yield cirq.Z(control)**(0.5 * self.half_turns)

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        return cirq.TextDiagramInfo(
            wire_symbols=('@', 'XXYY', 'XXYY'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'CXXYY'
        return 'CXXYY**{!r}'.format(self.half_turns)


class ControlledYXXYGate(cirq.EigenGate,
                         cirq.CompositeGate,
                         cirq.TextDiagrammable):
    """Controlled YX - XY interaction."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:

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
        return ControlledYXXYGate(half_turns=exponent)

    def default_decompose(self, qubits):
        control, a, b = qubits
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=5/8).on(a)
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=7/8).on(b)
        yield cirq.Y(a)**-0.5, cirq.Y(b)**-0.5
        yield CCZ(control, a, b)**self.half_turns
        yield cirq.CZ(control, a)**(-0.5 * self.half_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.half_turns)
        yield cirq.Y(a)**0.5, cirq.Y(b)**0.5
        yield cirq.X(a)**0.5, cirq.X(b)**0.5
        yield CCZ(control, a, b)**self.half_turns
        yield cirq.CZ(control, a)**(-0.5 * self.half_turns)
        yield cirq.CZ(control, b)**(-0.5 * self.half_turns)
        yield cirq.X(a)**-0.5, cirq.X(b)**-0.5
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=5/8).on(a)
        yield cirq.google.ExpWGate(half_turns=1, axis_half_turns=7/8).on(b)
        yield cirq.Z(control)**(0.5 * self.half_turns)

    def text_diagram_info(self, args: cirq.TextDiagramInfoArgs
                          ) -> cirq.TextDiagramInfo:
        return cirq.TextDiagramInfo(
            wire_symbols=('@', 'YXXY', '#2'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'CYXXY'
        return 'CYXXY**{!r}'.format(self.half_turns)


CCZ = Rot111Gate()
CXXYY = ControlledXXYYGate()
CYXXY = ControlledYXXYGate()
