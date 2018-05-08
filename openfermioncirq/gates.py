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
from typing import Union

import numpy

import cirq
from cirq.ops.partial_reflection_gate import PartialReflectionGate


def _canonicalize_half_turns(half_turns: float) -> float:
    half_turns %= 4
    if half_turns > 2:
        half_turns -= 4
    return half_turns


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
               cirq.ExtrapolatableGate,
               cirq.InterchangeableQubitsGate,
               cirq.KnownMatrixGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """XX + YY interaction.

    This gate implements the unitary exp(-i pi half_turns (XX + YY) / 4)
    """

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def matrix(self):
        c = numpy.cos(numpy.pi * self.half_turns / 2)
        s = numpy.sin(numpy.pi * self.half_turns / 2)
        return numpy.array([[1, 0, 0, 0],
                            [0, c, -1j * s, 0],
                            [0, -1j * s, c, 0],
                            [0, 0, 0, 1]])

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

    def extrapolate_effect(self, factor) -> 'XXYYGate':
        return XXYYGate(half_turns=self.half_turns * factor)

    def inverse(self) -> 'XXYYGate':
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, XXYYGate):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((XXYYGate, self.half_turns))

    def __repr__(self):
        return 'XXYYGate(half_turns={!r})'.format(self.half_turns)


class YXXYGate(cirq.CompositeGate,
               cirq.ExtrapolatableGate,
               cirq.KnownMatrixGate,
               cirq.TextDiagrammableGate,
               cirq.TwoQubitGate):
    """YX - XY interaction.

    This gate implements the unitary exp(-i pi half_turns (YX - XY) / 4)
    """

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def matrix(self):
        c = numpy.cos(numpy.pi * self.half_turns / 2)
        s = numpy.sin(numpy.pi * self.half_turns / 2)
        return numpy.array([[1, 0, 0, 0],
                            [0, c, -s, 0],
                            [0, s, c, 0],
                            [0, 0, 0, 1]])

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

    def extrapolate_effect(self, factor) -> 'YXXYGate':
        return YXXYGate(half_turns=self.half_turns * factor)

    def inverse(self) -> 'YXXYGate':
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, YXXYGate):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((YXXYGate, self.half_turns))

    def __repr__(self):
        return 'YXXYGate(half_turns={!r})'.format(self.half_turns)


class Rot111Gate(cirq.CompositeGate,
                 cirq.InterchangeableQubitsGate,
                 PartialReflectionGate):
    """Phases the |111> state of three qubits by a fixed amount."""

    def _with_half_turns(self,
                         half_turns: Union[cirq.Symbol, float] = 1.0
                         ) -> 'Rot11Gate':
        return Rot111Gate(half_turns=half_turns)

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

    def _reflection_matrix(self):
        """See base class."""
        return numpy.diag([1, 1, 1, 1, 1, 1, 1, -1])

    def __str__(self):
        base = 'CCZ'
        if self.half_turns == 1:
            return base
        return '{}**{}'.format(base, repr(self.half_turns))

    def __repr__(self) -> str:
        return self.__str__()


class ControlledXXYYGate(cirq.CompositeGate,
                         cirq.ExtrapolatableGate,
                         cirq.KnownMatrixGate,
                         cirq.TextDiagrammableGate):
    """Controlled XX + YY interaction."""

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def matrix(self):
        return cirq.block_diag(numpy.eye(4), (XXYY**(self.half_turns)).matrix())

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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'XXYY', 'XXYY'

    def text_diagram_exponent(self):
        return self.half_turns

    def extrapolate_effect(self, factor) -> 'ControlledXXYYGate':
        return ControlledXXYYGate(half_turns=self.half_turns * factor)

    def inverse(self) -> 'ControlledXXYYGate':
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, ControlledXXYYGate):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ControlledXXYYGate, self.half_turns))

    def __repr__(self):
        return 'ControlledXXYYGate(half_turns={!r})'.format(self.half_turns)


class ControlledYXXYGate(cirq.CompositeGate,
                         cirq.ExtrapolatableGate,
                         cirq.KnownMatrixGate,
                         cirq.TextDiagrammableGate):
    """Controlled YX - XY interaction."""

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def matrix(self):
        return cirq.block_diag(numpy.eye(4), (YXXY**(self.half_turns)).matrix())

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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'YXXY', '#2'

    def text_diagram_exponent(self):
        return self.half_turns

    def extrapolate_effect(self, factor) -> 'ControlledYXXYGate':
        return ControlledYXXYGate(half_turns=self.half_turns * factor)

    def inverse(self) -> 'ControlledYXXYGate':
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, ControlledYXXYGate):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ControlledYXXYGate, self.half_turns))

    def __repr__(self):
        return 'ControlledYXXYGate(half_turns={!r})'.format(self.half_turns)


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
YXXY = YXXYGate()
CCZ = Rot111Gate()
CXXYY = ControlledXXYYGate()
CYXXY = ControlledYXXYGate()
