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
from typing import Sequence, Tuple, Union

import numpy

import cirq


def iterative_phase_estimation(system_qubits: Sequence[cirq.QubitId],
                               ancilla_qubit: cirq.QubitId,
                               controlled_unitaries: Sequence[cirq.OP_TREE],
                               preparation_ops: cirq.OP_TREE=(),
                               initial_state: Union[int, numpy.ndarray]=0,
                               repetitions: Union[int, Sequence[int]]=1,
                               simulator: cirq.google.Simulator=None,
                               ) -> Tuple[float, numpy.ndarray]:
    """Perform iterative phase estimation.

    This is a method for estimating an eigenvalue e^{2 pi phi} of a unitary U.
    It outputs an estimate of the phase phi along with an approximate
    eigenstate. The inputs are a list of operations for performing
    controlled-U^{2^k}, where k ranges from 0 to m-1, where m is the desired
    number of bits of precision. One should also input an initial state and/or
    operations used to prepare an initial state. If the initial state has a
    large (over 50%) overlap with an eigenstate of U, then with reasonably
    high probability, that eigenstate and its phase will be returned. The
    probability of this result occurring can be amplified by requesting that
    certain bits of the phase be measured to a higher accuracy through the
    `repetitions` argument. Since the less significant bits contribute more
    to the failure probbility, they should be prioritized; for instance,
    a reasonable value for `repetitions` would be [17, 9, 5, 3, 1, 1, 1, 1].

    Implementation based on arXiv:quant-ph/0610214.

    Args:
        controlled_unitaries: A list of operations that act on the given ancilla
            and system qubits, where the ancilla is treated as the control
            qubit. The k-th OP_TREE in the list performs controlled-U^{2^k},
            where U is the unitary whose eigenvalue is to be estimated.
        initial_state: The initial state of the system qubits. Must be safely
            castable to dtype numpy.complex64.
        preparation_ops: Any operations that should be applied to the initial
            state of the system qubits.
        repetitions: If an integer, specifies the number of times to measure
            each bit of the output before moving on to the next stage. If a
            sequence of integers, then the k-th integer specifies the number of
            times to measure the k-th least significant bit before moving on.
        simulator: The simulator to use.
    """
    # TODO use reasonable default for repetitions parameter.
    # maybe take in error probability instead

    simulator = simulator or cirq.google.XmonSimulator()
    n_bits = len(controlled_unitaries)

    # Prepare initial state
    preparation_circuit = cirq.Circuit.from_ops(preparation_ops)
    result = simulator.run(preparation_circuit,
                           qubit_order=system_qubits,
                           initial_state=initial_state)
    zero = [1, 0]
    initial_state = numpy.kron(zero, result.final_state).astype(
            numpy.complex64)

    last_measured_bit = 0
    current_state = initial_state
    feedback_quarter_turns = 0.0
    bits_of_phase = []

    for i in range(n_bits):
        # Determine the i-th least significant bit of the phase

        k = n_bits - 1 - i
        controlled_unitary = controlled_unitaries[k]
        reps = repetitions if isinstance(repetitions, int) else repetitions[i]
        feedback_half_turns = 2 * feedback_quarter_turns

        # If the last measured bit was 1, flip it back to 0
        ancilla_flip = cirq.X(ancilla_qubit) if last_measured_bit == 1 else ()

        # Measure the bit
        circuit = cirq.Circuit.from_ops(
                ancilla_flip,
                _measure_bit_of_phase(ancilla_qubit,
                                      system_qubits,
                                      controlled_unitary,
                                      feedback_half_turns))
        result = simulator.run(
            circuit,
            qubit_order=[ancilla_qubit] + list(system_qubits),
            initial_state=current_state,
            repetitions=reps)

        # Determine the bit by majority vote
        num_ones = numpy.count_nonzero(result.measurements['ancilla_qubit'])
        num_zeros = reps - num_ones
        last_measured_bit = num_ones > num_zeros

        # Get one of the resulting states with the correct bit measured
        current_state = result.final_states[
                numpy.where(result.measurements['ancilla_qubit'] ==
                            last_measured_bit)[0][0]]

        # Update the feedback angle
        feedback_quarter_turns /= 2
        feedback_quarter_turns -= last_measured_bit / 4

        bits_of_phase.append(last_measured_bit)

    # Compute phase
    phase = sum(bits_of_phase[i] / 2**(n_bits - i) for i in range(n_bits))

    # Get state of system
    dim = 2**len(system_qubits)
    if last_measured_bit == 0:
        result_state = current_state[:dim]
    else:
        result_state = current_state[dim:]

    return phase, result_state


def _measure_bit_of_phase(ancilla_qubit: cirq.QubitId,
                          system_qubits: Sequence[cirq.QubitId],
                          controlled_unitary: cirq.OP_TREE,
                          feedback_half_turns: float) -> cirq.OP_TREE:
    yield cirq.H(ancilla_qubit)
    yield controlled_unitary
    yield cirq.Z(ancilla_qubit)**feedback_half_turns
    yield cirq.H(ancilla_qubit)
    yield cirq.MeasurementGate('ancilla_qubit').on(ancilla_qubit)
