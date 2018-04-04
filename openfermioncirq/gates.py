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

import cirq


def _canonicalize_half_turns(half_turns: float) -> float:
    half_turns %= 2
    if half_turns > 1:
        half_turns -= 2
    return half_turns


class FermionicSwapGate(cirq.AsciiDiagrammableGate,
                        cirq.CompositeGate,
                        cirq.InterchangeableQubitsGate,
                        cirq.KnownMatrixGate,
                        cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def matrix(self):
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])

    def ascii_wire_symbols(self):
        return '×ᶠ', '×ᶠ'

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.SWAP(a, b)
        yield cirq.CZ(a, b)

    def __repr__(self):
        return 'FSWAP'


class XXYYGate(cirq.AsciiDiagrammableGate,
               cirq.ExtrapolatableGate,
               cirq.InterchangeableQubitsGate,
               cirq.KnownMatrixGate,
               cirq.TwoQubitGate):
    """XX + YY interaction."""

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def matrix(self):
        c = numpy.cos(numpy.pi * self.half_turns)
        s = numpy.sin(numpy.pi * self.half_turns)
        return numpy.array([[1, 0, 0, 0],
                            [0, c, -1j * s, 0],
                            [0, -1j * s, c, 0],
                            [0, 0, 0, 1]])

    def ascii_wire_symbols(self):
        return 'XXYY', 'XXYY'

    def ascii_exponent(self):
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


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
