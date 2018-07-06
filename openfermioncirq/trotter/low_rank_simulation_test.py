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

import os

import numpy
import scipy.sparse.linalg

import cirq

from openfermion.config import THIS_DIRECTORY
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, get_sparse_operator
from openfermion.utils import count_qubits, random_interaction_operator

from openfermioncirq.primitives import swap_network
from openfermioncirq.trotter.low_rank_simulation import low_rank_trotter_step


def fidelity(state1, state2):
    return abs(numpy.dot(state1, numpy.conjugate(state2))) ** 2

def test_low_rank_simulation_random():

    # Simulation parameters.
    seed = 1
    n_qubits = 4
    n_steps = 10
    time_scale = 20.

    # Initialize a random two-body FermionOperator.
    qubits = cirq.LineQubit.range(n_qubits)
    random_io = random_interaction_operator(n_qubits, seed=seed)
    random_operator = get_fermion_operator(random_io)
    time = time_scale / random_operator.induced_norm()

    # Construct Cirq circuit.
    trotter_time = time / float(n_steps)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit.from_ops(
        low_rank_trotter_step(qubits, trotter_time * random_operator))
    if n_qubits % 2 == 0:
        circuit.append(swap_network(qubits))
    circuit *= n_steps

    # Construct a random initial state.
    numpy.random.seed(seed)
    initial_state = numpy.random.randn(2 ** n_qubits)
    initial_state /= numpy.linalg.norm(initial_state)
    initial_state = initial_state.astype(numpy.complex64, copy=False)
    assert numpy.allclose(numpy.linalg.norm(initial_state), 1.0)

    # Simulate exact evolution.
    hamiltonian_sparse = get_sparse_operator(random_operator)
    correct_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * hamiltonian_sparse, initial_state)

    # Simulate Cirq circuit.
    simulator = cirq.google.XmonSimulator()
    result = simulator.simulate(circuit,
                                qubit_order=qubits,
                                initial_state=initial_state)

    # Check fidelity.
    final_state = result.final_state
    initial_fidelity = fidelity(initial_state, correct_state)
    final_fidelity = fidelity(final_state, correct_state)
    print(initial_fidelity, final_fidelity)
    assert initial_fidelity < 0.9
    assert final_fidelity > 0.99

def test_low_rank_simulation_h2():

    # Simulation parameters.
    n_steps = 10
    time_scale = 20.

    # Initialize molecule.
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
    basis = 'sto-3g'
    multiplicity = 1
    filename = os.path.join(THIS_DIRECTORY, 'data',
                            'H2_sto-3g_singlet_0.7414')
    molecule = MolecularData(
        geometry, basis, multiplicity, filename=filename)
    molecule.load()

    # Get molecular Hamiltonian.
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    # Get fermion Hamiltonian.
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

    # Initialize a random two-body FermionOperator.
    n_qubits = count_qubits(fermion_hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)
    time = time_scale / fermion_hamiltonian.induced_norm()

    # Construct Cirq circuit.
    trotter_time = time / float(n_steps)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit.from_ops(
        low_rank_trotter_step(qubits, fermion_hamiltonian, time=trotter_time))
    circuit.append(swap_network(qubits))
    circuit *= n_steps

    # Construct a random initial state.
    initial_state = numpy.random.randn(2 ** n_qubits)
    initial_state /= numpy.linalg.norm(initial_state)
    initial_state = initial_state.astype(numpy.complex64, copy=False)
    assert numpy.allclose(numpy.linalg.norm(initial_state), 1.0)

    # Simulate exact evolution.
    hamiltonian_sparse = get_sparse_operator(fermion_hamiltonian)
    correct_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * hamiltonian_sparse, initial_state)

    # Simulate Cirq circuit.
    simulator = cirq.google.XmonSimulator()
    result = simulator.simulate(circuit,
                                qubit_order=qubits,
                                initial_state=initial_state)

    # Check fidelity.
    final_state = result.final_state
    initial_fidelity = fidelity(initial_state, correct_state)
    final_fidelity = fidelity(final_state, correct_state)
    print(initial_fidelity, final_fidelity)
    assert initial_fidelity < 0.9
    assert final_fidelity > 0.99
