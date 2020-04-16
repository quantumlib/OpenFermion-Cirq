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

import abc
from typing import Optional, Sequence, Tuple, Union

import cirq
import numpy as np
import openfermion
import scipy.linalg as la
import sympy


def _arg(x):
    if x == 0:
        return 0
    if cirq.is_parameterized(x):
        return sympy.arg(x)
    return np.angle(x)


def _canonicalize_weight(w):
    if w == 0:
        return (0, 0)
    if cirq.is_parameterized(w):
        return (cirq.PeriodicValue(abs(w), 2 * sympy.pi), sympy.arg(w))
    period = 2 * np.pi
    return (np.round((w.real % period) if (w == np.real(w)) else
                     (abs(w) % period) * w / abs(w), 8), 0)


def state_swap_eigen_component(x: str, y: str, sign: int = 1, angle: float = 0):
    """The +/- eigen-component of the operation that swaps states x and y.

    For example, state_swap_eigen_component('01', '10', ±1) with angle θ returns
        ┌                               ┐
        │0, 0,           0,            0│
        │0, 0.5,         ±0.5 e^{-iθ}, 0│
        │0, ±0.5 e^{iθ}, 0.5,          0│
        │0, 0,           0,            0│
        └                               ┘

    Args:
        x: The first state to swap, as a bitstring.
        y: The second state to swap, as a bitstring. Must have high index than
            x.
        sign: The sign of the off-diagonal elements (indicated by +/-1).
        angle: The phase of the complex off-diagonal elements. Defaults to 0.

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

    dim = 2**len(x)
    i, j = int(x, 2), int(y, 2)

    component = np.zeros((dim, dim), dtype=np.complex128)
    component[i, i] = component[j, j] = 0.5
    component[j, i] = sign * 0.5 * 1j**(angle * 2 / np.pi)
    component[i, j] = sign * 0.5 * 1j**(-angle * 2 / np.pi)
    return component


@cirq.value_equality(approximate=True)
class ParityPreservingFermionicGate(cirq.Gate, metaclass=abc.ABCMeta):
    r"""The Jordan-Wigner transform of :math:`\exp(-i H)` for a fermionic
    Hamiltonian :math:`H`.

    Each subclass corresponds to a set of generators :math:`\{G_i\}`
    corresponding to the family of Hamiltonians :math:`\sum_i w_i G_i +
    \text{h.c.}`, where the weights :math:`w_i \in \mathbb C` are specified by
    the instance.

    The Jordan-Wigner mapping maps the fermionic modes :math:`(0, \ldots, n -
    1)` to qubits :math:`(0, \ldots, n - 1)`, respectively.

    Each generator :math:`G_i` must be a linear combination of fermionic
    monomials consisting of an even number of creation/annihilation operators.
    This is so that the Jordan-Wigner transform acts only on the gate's qubits,
    even when the fermionic modes are offset as part of a larger Jordan-Wigner
    string.
    """

    def __init__(
            self,
            weights: Optional[Tuple[complex, ...]] = None,
            absorb_exponent: bool = False,
            exponent: cirq.TParamVal = 1.0,
            global_shift: float = 0.0,
    ) -> None:
        """A fermionic interaction.

        Args:
            weights: The weights of the terms in the Hamiltonian.
            absorb_exponent: Whether to absorb the given exponent into the
                weights. If true, the exponent of the return gate is `1`.
                Defaults to `False`.
        """
        if weights is None:
            weights = (1.,) * self.num_weights
        self.weights = weights

        self._exponent = exponent
        self._global_shift = global_shift
        self._canonical_exponent_cached = None

        if absorb_exponent:
            self.absorb_exponent_into_weights()

    @abc.abstractproperty
    def fermion_generator_components(self
                                    ) -> Tuple[openfermion.FermionOperator]:
        r"""The FermionOperators :math:`(G_i)_i` such that the gate's fermionic
        generator is :math:`\sum_i w_i G_i + \text{h.c.}` where :math:`(w_i)_i`
        are the gate's weights."""

    @abc.abstractmethod
    def fswap(self, i: int):
        """Update the weights of the gate to effect conjugation by an FSWAP on
        the i-th and (i+1)th qubits."""

    @property
    def num_weights(self) -> int:
        """The number of parameters (weights) in the generator."""
        return len(self.fermion_generator_components)

    @property
    def qubit_generator_matrix(self) -> np.ndarray:
        """The matrix G such that the gate's unitary is exp(-i t G) with
        exponent t."""
        return openfermion.jordan_wigner_sparse(self.fermion_generator,
                                                self.num_qubits()).toarray()

    @property
    def fermion_generator(self) -> openfermion.FermionOperator:
        """The FermionOperator G such that the gate's unitary is exp(-i t G)
        with exponent t using the Jordan-Wigner transformation."""
        half_generator = sum(
            (w * G
             for w, G in zip(self.weights, self.fermion_generator_components)),
            openfermion.FermionOperator())
        return half_generator + openfermion.hermitian_conjugated(half_generator)

    def _value_equality_values_(self):
        return tuple(
            _canonicalize_weight(w * self.exponent)
            for w in list(self.weights) + [self._global_shift])

    def _is_parameterized_(self) -> bool:
        return any(
            cirq.is_parameterized(v)
            for V in self._value_equality_values_()
            for v in V)

    def absorb_exponent_into_weights(self):
        period = (2 * sympy.pi) if self._is_parameterized_() else 2 * (np.pi)
        new_weights = []
        for weight in self.weights:
            if not weight:
                new_weights.append(weight)
                continue
            old_abs = abs(weight)
            new_abs = (old_abs * self._exponent) % period
            new_weights.append(weight * new_abs / old_abs)
        self.weights = tuple(new_weights)
        self._global_shift *= self._exponent
        self._exponent = 1

    def permute(self, init_pos: Sequence[int]):
        """An in-place version of permuted."""
        I = range(self.num_qubits())
        if sorted(init_pos) != list(I):
            raise ValueError(f'{init_pos} is not a permutation of {I}.')
        curr_pos = list(init_pos)
        for i in I:
            for j in I[i % 2:-1:2]:
                if curr_pos[j] > curr_pos[j + 1]:
                    self.fswap(j)
                    curr_pos[j:j + 2] = reversed(curr_pos[j:j + 2])
        assert curr_pos == list(I)

    def permuted(self, init_pos: Sequence[int]):
        """Returns a gate with the Jordan-Wigner ordering changed.

        If the Jordan-Wigner ordering of the original gate is given by
        init_pos, then the returned gate has Jordan-Wigner ordering
        (0, ..., n - 1), where n is the number of qubits on which the gate acts.

        Args:
            init_pos: A permutation of (0, ..., n - 1).
        """
        gate = self.__copy__()
        gate.permute(init_pos)
        return gate

    def __copy__(self):
        return type(self)(self.weights,
                          exponent=self.exponent,
                          global_shift=self._global_shift)


class QuadraticFermionicSimulationGate(ParityPreservingFermionicGate,
                                       cirq.InterchangeableQubitsGate,
                                       cirq.TwoQubitGate, cirq.EigenGate):
    r"""(w0 |10><01| + h.c.) + w1 * |11><11| interaction.

    With weights :math:`(w_0, w_1)` and exponent :math:`t`, this gate's matrix
    is defined as

    .. math::
        e^{-i t H},

    where

    .. math::
        H = \left(w_0 \left| 10 \right\rangle\left\langle 01 \right| +
                \text{h.c.}\right) -
            w_1 \left| 11 \right\rangle \left\langle 11 \right|.

    This corresponds to the Jordan-Wigner transform of

    .. math::
        H = (w_0 a^{\dagger}_i a_{i+1} + \text{h.c.}) +
             w_1 a_{i}^{\dagger} a_{i+1}^{\dagger} a_{i} a_{i+1},

    where :math:`a_i` and  :math:`a_{i+1}` are the annihilation operators for
    the fermionic modes :math:`i` and :math:`(i+1)`, respectively mapped to the
    first and second qubits on which this gate acts.

    Args:
        weights: The weights of the terms in the Hamiltonian.
    """

    @property
    def num_weights(self):
        return 2

    def _decompose_(self, qubits):
        r = 2 * abs(self.weights[0]) / np.pi
        theta = _arg(self.weights[0]) / np.pi
        yield cirq.Z(qubits[0])**-theta
        yield cirq.ISwapPowGate(exponent=-r * self.exponent)(*qubits)
        yield cirq.Z(qubits[0])**theta
        yield cirq.CZPowGate(exponent=-self.weights[1] * self.exponent /
                             np.pi)(*qubits)

    def _eigen_components(self):
        components = [(0, np.diag([1, 0, 0, 0])),
                      (-self.weights[1] / np.pi, np.diag([0, 0, 0, 1]))]
        r = abs(self.weights[0]) / np.pi
        theta = 2 * _arg(self.weights[0]) / np.pi
        for s in (-1, 1):
            components.append(
                (-s * r,
                 np.array([[0, 0, 0, 0], [0, 1, s * 1j**(-theta), 0],
                           [0, s * 1j**(theta), 1, 0], [0, 0, 0, 0]]) / 2))
        return components

    def __repr__(self):
        exponent_str = ('' if self.exponent == 1 else ', exponent=' +
                        cirq._compat.proper_repr(self.exponent))
        return ('ofc.QuadraticFermionicSimulationGate(({}){})'.format(
            ', '.join(cirq._compat.proper_repr(v) for v in self.weights),
            exponent_str))

    @property
    def qubit_generator_matrix(self):
        generator = np.zeros((4, 4), dtype=np.complex128)
        # w0 |10><01| + h.c.
        generator[2, 1] = self.weights[0]
        generator[1, 2] = self.weights[0].conjugate()
        # w1 |11><11|
        generator[3, 3] = self.weights[1]
        return generator

    @property
    def fermion_generator_components(self):
        return (
            openfermion.FermionOperator(((0, 1), (1, 0))),
            openfermion.FermionOperator(((0, 1), (0, 0), (1, 1), (1, 0)), 0.5),
        )

    def fswap(self, i: int = 0):
        if i != 0:
            raise ValueError(f'{i} != 0')
        self.weights = (self.weights[0].conjugate(), self.weights[1])


class CubicFermionicSimulationGate(ParityPreservingFermionicGate,
                                   cirq.ThreeQubitGate, cirq.EigenGate):
    r"""w0 * |110><101| + w1 * |110><011| + w2 * |101><011| + hc interaction.

    With weights :math:`(w_0, w_1, w_2)` and exponent :math:`t`, this gate's
    matrix is defined as

    .. math::
        e^{-i t H},

    where

    .. math::
        H = \left(w_0 \left| 110 \right\rangle\left\langle 101 \right| +
                \text{h.c.}\right) +
            \left(w_1 \left| 110 \right\rangle\left\langle 011 \right| +
                \text{h.c.}\right) +
            \left(w_2 \left| 101 \right\rangle\left\langle 011 \right| +
                \text{h.c.}\right)

    This corresponds to the Jordan-Wigner transform of

    .. math::
        H = -\left(w_0 a^{\dagger}_i a^{\dagger}_{i+1} a_{i} a_{i+2} +
                   \text{h.c.}\right) -
            \left(w_1 a^{\dagger}_i a^{\dagger}_{i+1} a_{i+1} a_{i+2} +
                  \text{h.c.}\right) -
            \left(w_2 a^{\dagger}_i a^{\dagger}_{i+2} a_{i+1} a_{i+2} +
                  \text{h.c.}\right),

    where :math:`a_i`, :math:`a_{i+1}`, :math:`a_{i+2}` are the annihilation
    operators for the fermionic modes :math:`i`, :math:`(i+1)` :math:`(i+2)`,
    respectively mapped to the three qubits on which this gate acts.

    Args:
        weights: The weights of the terms in the Hamiltonian.
    """

    @property
    def num_weights(self):
        return 3

    def _eigen_components(self):
        components = [(0, np.diag([1, 1, 1, 0, 1, 0, 0, 1]))]
        nontrivial_part = np.zeros((3, 3), dtype=np.complex128)
        for ij, w in zip([(1, 2), (0, 2), (0, 1)], self.weights):
            nontrivial_part[ij] = w
            nontrivial_part[ij[::-1]] = w.conjugate()
        assert np.allclose(nontrivial_part, nontrivial_part.conj().T)
        eig_vals, eig_vecs = np.linalg.eigh(nontrivial_part)
        for eig_val, eig_vec in zip(eig_vals, eig_vecs.T):
            exp_factor = -eig_val / np.pi
            proj = np.zeros((8, 8), dtype=np.complex128)
            nontrivial_indices = np.array([3, 5, 6], dtype=np.intp)
            proj[nontrivial_indices[:, np.newaxis], nontrivial_indices] = (
                np.outer(eig_vec.conjugate(), eig_vec))
            components.append((exp_factor, proj))
        return components

    def __repr__(self):
        return ('ofc.CubicFermionicSimulationGate(' + '({})'.format(' ,'.join(
            cirq._compat.proper_repr(w) for w in self.weights)) +
                ('' if self.exponent == 1 else
                 (', exponent=' + cirq._compat.proper_repr(self.exponent))) +
                ('' if self._global_shift == 0 else
                 (', global_shift=' +
                  cirq._compat.proper_repr(self._global_shift))) + ')')

    @property
    def qubit_generator_matrix(self):
        generator = np.zeros((8, 8), dtype=np.complex128)
        # w0 |110><101| + h.c.
        generator[6, 5] = self.weights[0]
        generator[5, 6] = self.weights[0].conjugate()
        # w1 |110><011| + h.c.
        generator[6, 3] = self.weights[1]
        generator[3, 6] = self.weights[1].conjugate()
        # w2 |101><011| + h.c.
        generator[5, 3] = self.weights[2]
        generator[3, 5] = self.weights[2].conjugate()
        return generator

    @property
    def fermion_generator_components(self):
        return (
            openfermion.FermionOperator(((0, 1), (0, 0), (1, 1), (2, 0))),
            openfermion.FermionOperator(((0, 1), (1, 1), (1, 0), (2, 0)), -1),
            openfermion.FermionOperator(((0, 1), (1, 0), (2, 1), (2, 0))),
        )

    def fswap(self, i: int):
        if i == 0:
            self.weights = (-self.weights[1], -self.weights[0],
                            self.weights[2].conjugate())
        elif i == 1:
            self.weights = (self.weights[0].conjugate(), -self.weights[2],
                            -self.weights[1])
        else:
            raise ValueError(f'{i} not in (0, 1)')


class QuarticFermionicSimulationGate(ParityPreservingFermionicGate,
                                     cirq.EigenGate):
    r"""Rotates Hamming-weight 2 states into their bitwise complements.

    With weights :math:`(w_0, w_1, w_2)` and exponent :math:`t`, this gate's
    matrix is defined as

    .. math::
        e^{-i t H},

    where

    .. math::
        H = \left(w_0 \left| 1001 \right\rangle\left\langle 0110 \right| +
                \text{h.c.}\right) +
            \left(w_1 \left| 1010 \right\rangle\left\langle 0101 \right| +
                \text{h.c.}\right) +
            \left(w_2 \left| 1100 \right\rangle\left\langle 0011 \right| +
                \text{h.c.}\right)

    This corresponds to the Jordan-Wigner transform of

    .. math::
        H = -\left(w_0 a^{\dagger}_i a^{\dagger}_{i+3} a_{i+1} a_{i+2} +
                   \text{h.c.}\right) -
            \left(w_1 a^{\dagger}_i a^{\dagger}_{i+2} a_{i+1} a_{i+3} +
                  \text{h.c.}\right) -
            \left(w_2 a^{\dagger}_i a^{\dagger}_{i+1} a_{i+2} a_{i+3} +
                  \text{h.c.}\right),

    where :math:`a_i`, ..., :math:`a_{i+3}` are the annihilation operators for
    the fermionic modes :math:`i`, ..., :math:`(i+3)`, respectively
    mapped to the four qubits on which this gate acts.


    Args:
        weights: The weights of the terms in the Hamiltonian.
    """

    @property
    def num_weights(self):
        return 3

    def num_qubits(self):
        return 4

    def _eigen_components(self):
        # projector onto subspace spanned by basis states with
        # Hamming weight != 2
        zero_component = np.diag(
            [int(bin(i).count('1') != 2) for i in range(16)])

        state_pairs = (('0110', '1001'), ('0101', '1010'), ('0011', '1100'))

        plus_minus_components = tuple(
            (-abs(weight) * sign / np.pi,
             state_swap_eigen_component(state_pair[0], state_pair[1], sign,
                                        np.angle(weight)))
            for weight, state_pair in zip(self.weights, state_pairs)
            for sign in (-1, 1))

        return ((0, zero_component),) + plus_minus_components

    def _with_exponent(self, exponent: Union[sympy.Symbol, float]
                      ) -> 'QuarticFermionicSimulationGate':
        gate = QuarticFermionicSimulationGate(self.weights)
        gate._exponent = exponent
        return gate

    def _decompose_(self, qubits):
        """The goal is to effect a rotation around an axis in the XY plane in
        each of three orthogonal 2-dimensional subspaces.

        First, the following basis change is performed:
            0000 ↦ 0001        0001 ↦ 1111
            1111 ↦ 0010        1110 ↦ 1100
                               0010 ↦ 0000
            0110 ↦ 0101        1101 ↦ 0011
            1001 ↦ 0110        0100 ↦ 0100
            1010 ↦ 1001        1011 ↦ 0111
            0101 ↦ 1010        1000 ↦ 1000
            1100 ↦ 1101        0111 ↦ 1011
            0011 ↦ 1110

        Note that for each 2-dimensional subspace of interest, the first two
        qubits are the same and the right two qubits are different. The desired
        rotations thus can be effected by a complex-version of a partial SWAP
        gate on the latter two qubits, controlled on the first two qubits. This
        partial SWAP-like gate can  be decomposed such that it is parameterized
        solely by a rotation in the ZY plane on the third qubit. These are the
        `individual_rotations`; call them U0, U1, U2.

        To decompose the double controlled rotations, we use four other
        rotations V0, V1, V2, V3 (the `combined_rotations`) such that
            U0 = V3 · V1 · V0
            U1 = V3 · V2 · V1
            U2 = V2 · V0
        """

        if self._is_parameterized_():
            return NotImplemented

        individual_rotations = [
            la.expm(0.5j * self.exponent *
                    np.array([[np.real(w), 1j * s * np.imag(w)],
                              [-1j * s * np.imag(w), -np.real(w)]]))
            for s, w in zip([1, -1, -1], self.weights)
        ]

        combined_rotations = {}
        combined_rotations[0] = la.sqrtm(
            np.linalg.multi_dot([
                la.inv(individual_rotations[1]), individual_rotations[0],
                individual_rotations[2]
            ]))
        combined_rotations[1] = la.inv(combined_rotations[0])
        combined_rotations[2] = np.linalg.multi_dot([
            la.inv(individual_rotations[0]), individual_rotations[1],
            combined_rotations[0]
        ])
        combined_rotations[3] = individual_rotations[0]

        controlled_rotations = {
            i: cirq.ControlledGate(
                cirq.MatrixGate(combined_rotations[i], qid_shape=(2,)))
            for i in range(4)
        }

        a, b, c, d = qubits

        basis_change = list(
            cirq.flatten_op_tree([
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

        controlled_rotations = list(
            cirq.flatten_op_tree([
                controlled_rotations[0](b, c),
                cirq.CNOT(a, b), controlled_rotations[1](b, c),
                cirq.CNOT(b, a),
                cirq.CNOT(a, b), controlled_rotations[2](b, c),
                cirq.CNOT(a, b), controlled_rotations[3](b, c)
            ]))

        controlled_swaps = [
            [cirq.CNOT(c, d), cirq.H(c)],
            cirq.CNOT(d, c),
            controlled_rotations,
            cirq.CNOT(d, c),
            [cirq.inverse(op) for op in reversed(controlled_rotations)],
            [cirq.H(c), cirq.CNOT(c, d)],
        ]

        return [basis_change, controlled_swaps, basis_change[::-1]]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs
                              ) -> cirq.CircuitDiagramInfo:
        if args.use_unicode_characters:
            wire_symbols = ('⇊⇈',) * 4
        else:
            wire_symbols = ('a*a*aa',) * 4
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols,
                                       exponent=self._diagram_exponent(args))

    def _apply_unitary_(self,
                        args: cirq.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return NotImplemented

        am, bm, cm = (la.expm(-1j * self.exponent *
                              np.array([[0, w], [w.conjugate(), 0]]))
                      for w in self.weights)

        a1 = args.subspace_index(0b1001)
        b1 = args.subspace_index(0b0101)
        c1 = args.subspace_index(0b0011)

        a2 = args.subspace_index(0b0110)
        b2 = args.subspace_index(0b1010)
        c2 = args.subspace_index(0b1100)

        cirq.apply_matrix_to_slices(args.target_tensor,
                                    am,
                                    slices=[a1, a2],
                                    out=args.available_buffer)
        cirq.apply_matrix_to_slices(args.available_buffer,
                                    bm,
                                    slices=[b1, b2],
                                    out=args.target_tensor)
        return cirq.apply_matrix_to_slices(args.target_tensor,
                                           cm,
                                           slices=[c1, c2],
                                           out=args.available_buffer)

    def __repr__(self):
        return ('ofc.QuarticFermionicSimulationGate(({}), '
                'absorb_exponent=False, '
                'exponent={})'.format(
                    ', '.join(
                        cirq._compat.proper_repr(v) for v in self.weights),
                    cirq._compat.proper_repr(self.exponent)))

    @property
    def qubit_generator_matrix(self):
        """The (Hermitian) matrix G such that the gate's unitary is
        exp(-i * G).
        """

        generator = np.zeros((1 << 4,) * 2, dtype=np.complex128)

        # w0 |1001><0110| + h.c.
        generator[9, 6] = self.weights[0]
        generator[6, 9] = self.weights[0].conjugate()
        # w1 |1010><0101| + h.c.
        generator[10, 5] = self.weights[1]
        generator[5, 10] = self.weights[1].conjugate()
        # w2 |1100><0011| + h.c.
        generator[12, 3] = self.weights[2]
        generator[3, 12] = self.weights[2].conjugate()
        return generator

    @property
    def fermion_generator_components(self):
        return (
            openfermion.FermionOperator(((0, 1), (1, 0), (2, 0), (3, 1)), -1),
            openfermion.FermionOperator(((0, 1), (1, 0), (2, 1), (3, 0))),
            openfermion.FermionOperator(((0, 1), (1, 1), (2, 0), (3, 0)), -1),
        )

    def fswap(self, i: int):
        if i == 0:
            self.weights = (self.weights[1].conjugate(),
                            self.weights[0].conjugate(), -self.weights[2])
        elif i == 1:
            self.weights = (-self.weights[0], self.weights[2], self.weights[1])
        elif i == 2:
            self.weights = (self.weights[1], self.weights[0], -self.weights[2])
        else:
            raise ValueError(f'{i} not in (0, 1, 2)')
