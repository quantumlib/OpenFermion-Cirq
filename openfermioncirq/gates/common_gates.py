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

from typing import Optional, Union

import numpy as np

import cirq


def rot11(rads: float):
    """Phases the |11> state of two qubits by e^{i rads}."""
    return cirq.CZ**(rads / np.pi)


class FermionicSwapGate(cirq.EigenGate,
                        cirq.InterchangeableQubitsGate,
                        cirq.TwoQubitGate):
    """Swaps two adjacent fermionic modes under the JWT."""

    def __init__(self, *,  # Forces keyword args.
                 exponent: Union[cirq.Symbol, float] = 1.0) -> None:
        super().__init__(exponent=exponent)

    def _eigen_components(self):
        return [
            (0, np.array([[1, 0,   0,   0],
                          [0, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 0],
                          [0, 0,   0,   0]])),
            (1, np.array([[0,  0,    0,   0],
                          [0,  0.5, -0.5, 0],
                          [0, -0.5,  0.5, 0],
                          [0,  0,    0,   1]])),
        ]

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'FermionicSwapGate':
        return FermionicSwapGate(exponent=exponent)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        if self.exponent != 1:
            return None

        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        ii = args.subspace_index(0b11)
        args.available_buffer[oi] = args.target_tensor[oi]
        args.target_tensor[oi] = args.target_tensor[io]
        args.target_tensor[io] = args.available_buffer[oi]
        args.target_tensor[ii] *= -1
        return args.target_tensor

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            symbols = '×ᶠ', '×ᶠ'
        else:
            symbols = 'fswap', 'fswap'
        return cirq.CircuitDiagramInfo(
            wire_symbols=symbols,
            exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'FSWAP'
        return 'FSWAP**{!r}'.format(self.exponent)

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'ofc.FSWAP'
        return '(ofc.FSWAP**{!r})'.format(self.exponent)


class XXYYGate(cirq.EigenGate,
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
                 exponent: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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
        return [
            (0, np.diag([1, 0, 0, 1])),
            (-0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0, 0, 0]])),
            (+0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, -0.5, 0],
                             [0, -0.5, 0.5, 0],
                             [0, 0, 0, 0]]))
        ]

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        inner_matrix = cirq.unitary(cirq.Rx(self.exponent * np.pi))
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           inner_matrix,
                                           slices=[oi, io],
                                           out=args.available_buffer)

    def _with_exponent(self, exponent: Union[cirq.Symbol, float]) -> 'XXYYGate':
        return XXYYGate(exponent=exponent)

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** 0.5
        yield YXXY(a, b) ** self.exponent
        yield cirq.Z(a) ** -0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('XXYY', 'XXYY'),
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'XXYY'
        return 'XXYY**{!r}'.format(self.exponent)


class YXXYGate(cirq.EigenGate,
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
                 exponent: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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
        return [
            (0, np.diag([1, 0, 0, 1])),
            (-0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, -0.5j, 0],
                             [0, 0.5j, 0.5, 0],
                             [0, 0, 0, 0]])),
            (0.5, np.array([[0, 0, 0, 0],
                            [0, 0.5, 0.5j, 0],
                            [0, -0.5j, 0.5, 0],
                            [0, 0, 0, 0]]))
        ]

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        inner_matrix = cirq.unitary(cirq.Ry(-self.exponent * np.pi))
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           inner_matrix,
                                           slices=[oi, io],
                                           out=args.available_buffer)

    def _with_exponent(self, exponent: Union[cirq.Symbol, float]) -> 'YXXYGate':
        return YXXYGate(exponent=exponent)

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z(a) ** -0.5
        yield XXYY(a, b) ** self.exponent
        yield cirq.Z(a) ** 0.5

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('YXXY', '#2'),
            exponent=self._diagram_exponent(args))

    def __repr__(self):
        if self.exponent == 1:
            return 'YXXY'
        return 'YXXY**{!r}'.format(self.exponent)


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
                 exponent: Optional[Union[cirq.Symbol, float]]=None,
                 rads: Optional[float]=None,
                 degs: Optional[float]=None,
                 duration: Optional[float]=None) -> None:
        """Initializes the gate.

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

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs
                        ) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        global_phase = 1j**-self.exponent
        relative_phase = 1j**(2 * self.exponent)
        args.target_tensor *= global_phase
        oi = args.subspace_index(0b01)
        io = args.subspace_index(0b10)
        args.target_tensor[io] *= relative_phase
        args.target_tensor[oi] *= relative_phase
        return args.target_tensor

    def _eigen_components(self):
        return [
            (-0.5, np.diag([1, 0, 0, 1])),
            (0.5, np.diag([0, 1, 1, 0])),
        ]

    def _period(self) -> Optional[float]:
        return 2  # override 4

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]) -> 'ZZGate':
        return ZZGate(exponent=exponent)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('Z', 'Z'),
            exponent=self._diagram_exponent(args))

    def __repr__(self) -> str:
        if self.exponent == 1:
            return 'ZZ'
        return 'ZZ**{!r}'.format(self.exponent)


def Rxxyy(rads: float) -> XXYYGate:
    """Returns a gate with the matrix exp(-i rads (X⊗X + Y⊗Y) / 2)."""
    return XXYYGate(exponent=2 * rads / np.pi)


def Ryxxy(rads: float) -> YXXYGate:
    """Returns a gate with the matrix exp(-i rads (Y⊗X - X⊗Y) / 2)."""
    return YXXYGate(exponent=2 * rads / np.pi)


FSWAP = FermionicSwapGate()
XXYY = XXYYGate()
YXXY = YXXYGate()
ZZ = ZZGate()
