# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

import numpy

import cirq

from openfermioncirq import CXXYY, iterative_phase_estimation


# Construct exact eigenstates
zero = numpy.array([1, 0], dtype=numpy.complex64)
one = numpy.array([0, 1], dtype=numpy.complex64)
xxyy_eigenstate_a = (numpy.array([0, 1, 1, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))
xxyy_eigenstate_b = (numpy.array([0, 1, -1, 0], dtype=numpy.complex64) /
                     numpy.sqrt(2))

# Construct approximate eigenstates
overlap = .95
large_coeff = numpy.sqrt(overlap)
small_coeff = numpy.sqrt(1 - overlap)

close_to_one = large_coeff * one + small_coeff * zero
close_to_xxyy_eigenstate_a = (large_coeff * xxyy_eigenstate_a +
                              small_coeff * xxyy_eigenstate_b)
close_to_xxyy_eigenstate_b = (large_coeff * xxyy_eigenstate_b +
                              small_coeff * xxyy_eigenstate_a)


@pytest.mark.parametrize(
        'gate, phase_prefactor, initial_state, eigenstate, bits_of_precision, '
        'repetitions, atol', [
            # Exact eigenstate, full precision
            (cirq.CZ, 2, one, one, None, 1, 1e-7),
            (CXXYY, -4, xxyy_eigenstate_a, xxyy_eigenstate_a, None, 1, 5e-6),
            (CXXYY, 4, xxyy_eigenstate_b, xxyy_eigenstate_b, None, 1, 5e-6),
            # Exact eigenstate, less than full precision
            (cirq.CZ, 2, one, one, 4, 8, 1e-7),
            (CXXYY, -4, xxyy_eigenstate_a, xxyy_eigenstate_a, 4, [16, 8, 4, 2],
                5e-6),
            (CXXYY, 4, xxyy_eigenstate_b, xxyy_eigenstate_b, 4, [16, 8, 4, 2],
                5e-6),
            # Approximate eigenstate, full precision
            (cirq.CZ, 2, close_to_one, one, None, 1, 1e-7),
            (CXXYY, -4, close_to_xxyy_eigenstate_a, xxyy_eigenstate_a, None, 1,
                5e-6),
            (CXXYY, 4, close_to_xxyy_eigenstate_b, xxyy_eigenstate_b, None, 1,
                5e-6),
])
def test_iterative_phase_estimation(
        gate, phase_prefactor, initial_state, eigenstate, bits_of_precision,
        repetitions, atol):

    test_bits_of_phase = ['01101', '110011', '1011010', '01111110']

    ancilla = cirq.LineQubit(-1)
    numpy.random.seed(51250)

    for bitstring in test_bits_of_phase:
        n_bits = len(bitstring)
        phase = int(bitstring, 2) / 2**n_bits
        bits_of_precision_ = bits_of_precision or n_bits
        precision = 0 if bits_of_precision is None else 2**-bits_of_precision_

        n_sys_qubits = int(numpy.log2(len(eigenstate)))
        sys_qubits = cirq.LineQubit.range(n_sys_qubits)

        gate_power = phase_prefactor * phase
        controlled_unitaries = [
                gate(ancilla, *sys_qubits)**(gate_power * 2**k)
                for k in range(bits_of_precision_)]
        calculated_phase, result_state = iterative_phase_estimation(
                sys_qubits,
                ancilla,
                controlled_unitaries,
                initial_state=initial_state,
                repetitions=repetitions)

        numpy.testing.assert_allclose(calculated_phase, phase, atol=precision)
        # State should have been projected onto closest eigenstate
        cirq.testing.assert_allclose_up_to_global_phase(
                result_state, eigenstate, atol=atol)
