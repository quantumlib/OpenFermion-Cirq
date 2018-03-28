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


class XXYYGate(cirq.ExtrapolatableGate,
               cirq.KnownMatrixGate,
               cirq.TwoQubitGate):
    """XX + YY interaction."""

    def __init__(self, *positional_args,
                 half_turns: float=1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def ascii_wire_symbols(self):
        return 'XX+YY', '#2'

    def ascii_exponent(self):
        return self.half_turns

    def extrapolate_effect(self, factor) -> 'XXYYGate':
        return XXYYGate(half_turns=self.half_turns * factor)

    def inverse(self) -> 'XXYYGate':
        return self.extrapolate_effect(-1)

    def matrix(self):
        c = numpy.cos(numpy.pi * self.half_turns)
        s = numpy.sin(numpy.pi * self.half_turns)
        return numpy.array([[1, 0, 0, 0],
                            [0, c, -1j * s, 0],
                            [0, -1j * s, c, 0],
                            [0, 0, 0, 1]])

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


class FermionicSwapGate(cirq.CompositeGate,
                        cirq.KnownMatrixGate,
                        cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def matrix(self):
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1]])

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.SWAP(a, b)
        yield cirq.CZ(a, b)

    def __repr__(self):
        return 'FSWAP'


FSWAP = FermionicSwapGate()


class FermionicSimulationGate(cirq.AsciiDiagrammableGate,
                              cirq.CompositeGate,
                              cirq.InterchangeableQubitsGate,
                              cirq.KnownMatrixGate,
                              cirq.TwoQubitGate):
    """Fermionic simulation gate applied to adjacent fermionic modes."""

    def __init__(self,
                 kinetic_coeff: float=0.,
                 potential_coeff: float=0.) -> None:
        self.kinetic_angle = kinetic_coeff % (2. * numpy.pi)
        self.potential_angle = potential_coeff % (2. * numpy.pi)

    def ascii_wire_symbols(self):
        return str(self), '#2'

    def matrix(self):
        return numpy.array(
                [[1., 0., 0., 0.],
                 [0., -1.j * numpy.sin(self.kinetic_angle),
                  numpy.cos(self.kinetic_angle), 0.],
                 [0., numpy.cos(self.kinetic_angle),
                  -1.j * numpy.sin(self.kinetic_angle), 0.],
                 [0., 0., 0., -numpy.exp(-1.j * self.potential_angle)]])

    def default_decompose(self, qubits):
        a, b = qubits
        yield cirq.Rot11Gate(half_turns=self.potential_angle / numpy.pi)(a, b)
        yield XXYYGate(half_turns=self.kinetic_angle / numpy.pi)(a, b)
        yield FSWAP(a, b)

    def __str__(self):
        return 'F({}, {})'.format(self.kinetic_angle, self.potential_angle)

    def __repr__(self):
        return 'FermionicSimulationGate({}, {})'.format(
                self.kinetic_angle, self.potential_angle)
