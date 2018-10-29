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

from typing import Optional, Union, Sequence

import numpy as np

import cirq
from cirq.type_workarounds import NotImplementedType

from openfermioncirq.gates import common_gates


def rot111(rads: float):
    """Phases the |111> state of three qubits by e^{i rads}."""
    return cirq.CCZ**(rads / np.pi)


class ControlledXXYYGate(cirq.EigenGate,
                         cirq.ThreeQubitGate):
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
            exponent = 2 * duration / np.pi
        else:
            exponent = cirq.value.chosen_angle_to_half_turns(
                    half_turns=half_turns,
                    rads=rads,
                    degs=degs)

        super().__init__(exponent=exponent)

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        return cirq.apply_unitary_to_tensor(
            cirq.ControlledGate(common_gates.XXYY**self.half_turns),
            target_tensor,
            available_buffer,
            axes,
            default=NotImplemented)

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, 0.5],
                         [0.5, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, -0.5],
                         [-0.5, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component),
                (0.5, plus_half_component)]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'ControlledXXYYGate':
        return ControlledXXYYGate(half_turns=exponent)

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.H(a)
        yield cirq.CCZ(control, a, b)**self.half_turns
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.half_turns / 2)
        yield cirq.H(a)
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('@', 'XXYY', 'XXYY'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'CXXYY'
        return 'CXXYY**{!r}'.format(self.half_turns)


class ControlledYXXYGate(cirq.EigenGate,
                         cirq.ThreeQubitGate):
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
            exponent = 2 * duration / np.pi
        else:
            exponent = cirq.value.chosen_angle_to_half_turns(
                    half_turns=half_turns,
                    rads=rads,
                    degs=degs)

        super().__init__(exponent=exponent)

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        return cirq.apply_unitary_to_tensor(
            cirq.ControlledGate(common_gates.YXXY**self.half_turns),
            target_tensor,
            available_buffer,
            axes,
            default=NotImplemented)

    @property
    def half_turns(self) -> Union[cirq.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        minus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, -0.5j],
                         [0.5j, 0.5]]),
            np.diag([0]))
        plus_half_component = cirq.linalg.block_diag(
            np.diag([0, 0, 0, 0, 0]),
            np.array([[0.5, 0.5j],
                         [-0.5j, 0.5]]),
            np.diag([0]))

        return [(0, np.diag([1, 1, 1, 1, 1, 0, 0, 1])),
                (-0.5, minus_half_component),
                (0.5, plus_half_component)]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self,
                       exponent: Union[cirq.Symbol, float]
                       ) -> 'ControlledYXXYGate':
        return ControlledYXXYGate(half_turns=exponent)

    def _decompose_(self, qubits):
        control, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.X(a)**0.5
        yield cirq.CCZ(control, a, b)**self.half_turns
        # Note: Clifford optimization would merge this CZ into the CCZ decomp.
        yield cirq.CZ(control, b)**(-self.half_turns / 2)
        yield cirq.X(a)**-0.5
        yield cirq.CNOT(a, b)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                               ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=('@', 'YXXY', '#2'),
            exponent=self.half_turns)

    def __repr__(self):
        if self.half_turns == 1:
            return 'CYXXY'
        return 'CYXXY**{!r}'.format(self.half_turns)


CXXYY = ControlledXXYYGate()
CYXXY = ControlledYXXYGate()
