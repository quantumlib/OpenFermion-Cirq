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

import openfermion

from openfermioncirq.variational.ansatzes import SplitOperatorTrotterAnsatz


# Construct a Hubbard model Hamiltonian
hubbard_model = openfermion.fermi_hubbard(2, 2, 1., 4.)
hubbard_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(
        hubbard_model)

# Construct a jellium model Hamiltonian
grid = openfermion.Grid(2, 2, 1.0)
jellium = openfermion.jellium_model(grid, spinless=True, plane_wave=False)
jellium_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)

# Construct a Hamiltonian of ones
ones_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
        one_body=numpy.ones((5, 5)),
        two_body=numpy.ones((5, 5)))


def test_split_operator_trotter_ansatz_parameters():

    ansatz = SplitOperatorTrotterAnsatz(hubbard_hamiltonian)
    assert (set(ansatz.param_names()) ==
            set(ansatz.params.keys()) ==
            {symbol.name for symbol in ansatz.params.values()} ==
            {'U0', 'U1', 'U6', 'U7',
             'V0_1', 'V2_3', 'V4_5', 'V6_7'})

    ansatz = SplitOperatorTrotterAnsatz(hubbard_hamiltonian, iterations=2)
    assert (set(ansatz.param_names()) ==
            set(ansatz.params.keys()) ==
            {symbol.name for symbol in ansatz.params.values()} ==
            {'U0-0', 'U1-0', 'U6-0', 'U7-0',
             'V0_1-0', 'V2_3-0', 'V4_5-0', 'V6_7-0',
             'U0-1', 'U1-1', 'U6-1', 'U7-1',
             'V0_1-1', 'V2_3-1', 'V4_5-1', 'V6_7-1'})


def test_split_operator_trotter_ansatz_param_bounds():

    ansatz = SplitOperatorTrotterAnsatz(hubbard_hamiltonian)
    assert ansatz.param_names() == [
            'U0', 'U1', 'U6', 'U7',
            'V0_1', 'V2_3', 'V4_5', 'V6_7']
    assert ansatz.param_bounds() == [
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]


def test_split_operator_trotter_ansatz_circuit():

    complete_ansatz = SplitOperatorTrotterAnsatz(
            ones_hamiltonian,
            include_all_cz=True,
            include_all_z=True)
    circuit = complete_ansatz.circuit
    assert circuit.to_text_diagram(transpose=True).strip() == """
0     1           2           3            4
│     │           │           │            │
│     │           │           │            Z^0.0
│     │           │           │            │
│     │           │           YXXY─────────#2^0.5
│     │           │           │            │
│     │           │           Z^0.0        Z^0.0
│     │           │           │            │
│     │           YXXY────────#2^0.608     │
│     │           │           │            │
│     │           Z^0.0       YXXY─────────#2^-0.5
│     │           │           │            │
│     YXXY────────#2^-0.0187  Z^0.0        Z^0.0
│     │           │           │            │
│     Z^0.0       YXXY────────#2           │
│     │           │           │            │
YXXY──#2^-0.517   Z^0.0       YXXY─────────#2
│     │           │           │            │
Z     YXXY────────#2          Z^0.0        Z^0.0
│     │           │           │            │
Z^U0  Z^0.0       YXXY────────#2           │
│     │           │           │            │
Z     Z^U1        Z^0.0       YXXY─────────#2^-0.423
│     │           │           │            │
│     Z^0.0       Z^U2        Z^0.0        Z
│     │           │           │            │
│     │           Z^0.0       Z^U3         Z^U4
│     │           │           │            │
│     │           │           Z^0.0        Z
│     │           │           │            │
│     │           │           YXXY─────────#2^0.423
│     │           │           │            │
│     │           YXXY────────#2^-1        Z^0.0
│     │           │           │            │
│     YXXY────────#2^-1       Z^0.0        │
│     │           │           │            │
YXXY──#2^0.517    Z^0.0       YXXY─────────#2^-1
│     │           │           │            │
│     Z^0.0       YXXY────────#2^-1        Z^0.0
│     │           │           │            │
│     YXXY────────#2^0.0187   Z^0.0        │
│     │           │           │            │
@─────@^V0_1      Z^0.0       YXXY─────────#2^0.5
│     │           │           │            │
×─────×           YXXY────────#2^-0.608    Z^0.0
│     │           │           │            │
│     │           │           Z^0.0        │
│     │           │           │            │
│     │           │           YXXY─────────#2^-0.5
│     │           │           │            │
│     │           @───────────@^V2_3       Z^0.0
│     │           │           │            │
│     │           ×───────────×            │
│     │           │           │            │
│     @───────────@^V0_3      @────────────@^V2_4
│     │           │           │            │
│     ×───────────×           ×────────────×
│     │           │           │            │
@─────@^V1_3      @───────────@^V0_4       │
│     │           │           │            │
×─────×           ×───────────×            │
│     │           │           │            │
│     @───────────@^V1_4      @────────────@^V0_2
│     │           │           │            │
│     ×───────────×           ×────────────×
│     │           │           │            │
@─────@^V3_4      @───────────@^V1_2       │
│     │           │           │            │
×─────×           ×───────────×            │
│     │           │           │            │
Z^0.0 │           │           │            │
│     │           │           │            │
#2────YXXY^0.5    │           │            │
│     │           │           │            │
Z^0.0 Z^0.0       │           │            │
│     │           │           │            │
│     #2──────────YXXY^0.608  │            │
│     │           │           │            │
#2────YXXY^-0.5   Z^0.0       │            │
│     │           │           │            │
Z^0.0 Z^0.0       #2──────────YXXY^-0.0187 │
│     │           │           │            │
│     #2──────────YXXY        Z^0.0        │
│     │           │           │            │
#2────YXXY        Z^0.0       #2───────────YXXY^-0.517
│     │           │           │            │
Z^0.0 Z^0.0       #2──────────YXXY         Z
│     │           │           │            │
│     #2──────────YXXY        Z^0.0        Z^U0
│     │           │           │            │
#2────YXXY^-0.423 Z^0.0       Z^U1         Z
│     │           │           │            │
Z     Z^0.0       Z^U2        Z^0.0        │
│     │           │           │            │
Z^U4  Z^U3        Z^0.0       │            │
│     │           │           │            │
Z     Z^0.0       │           │            │
│     │           │           │            │
#2────YXXY^0.423  │           │            │
│     │           │           │            │
Z^0.0 #2──────────YXXY^-1     │            │
│     │           │           │            │
│     Z^0.0       #2──────────YXXY^-1      │
│     │           │           │            │
#2────YXXY^-1     Z^0.0       #2───────────YXXY^0.517
│     │           │           │            │
Z^0.0 #2──────────YXXY^-1     Z^0.0        │
│     │           │           │            │
│     Z^0.0       #2──────────YXXY^0.0187  │
│     │           │           │            │
#2────YXXY^0.5    Z^0.0       │            │
│     │           │           │            │
Z^0.0 #2──────────YXXY^-0.608 │            │
│     │           │           │            │
│     Z^0.0       │           │            │
│     │           │           │            │
#2────YXXY^-0.5   │           │            │
│     │           │           │            │
Z^0.0 │           │           │            │
│     │           │           │            │
""".strip()

    jellium_ansatz = SplitOperatorTrotterAnsatz(jellium_hamiltonian)
    circuit = jellium_ansatz.circuit
    assert circuit.to_text_diagram(transpose=True).strip() == """
0     1           2           3
│     │           │           │
│     │           │           Z^0.0
│     │           │           │
│     │           YXXY────────#2^0.5
│     │           │           │
│     │           Z^0.0       Z^0.0
│     │           │           │
│     YXXY────────#2^0.608    │
│     │           │           │
│     Z^0.0       YXXY────────#2^-0.333
│     │           │           │
YXXY──#2^0.667    Z^0.0       Z^0.0
│     │           │           │
Z     YXXY────────#2          │
│     │           │           │
Z     Z           YXXY────────#2^-0.392
│     │           │           │
│     Z^U1        Z^0.0       Z
│     │           │           │
│     Z           Z^U2        Z^U3
│     │           │           │
│     │           Z^0.0       Z
│     │           │           │
│     │           YXXY────────#2^0.392
│     │           │           │
│     YXXY────────#2^-1       Z^0.0
│     │           │           │
YXXY──#2^-0.667   Z^0.0       │
│     │           │           │
│     Z^0.0       YXXY────────#2^0.333
│     │           │           │
│     YXXY────────#2^-0.608   Z^0.0
│     │           │           │
@─────@^V0_1      Z^0.0       │
│     │           │           │
×─────×           YXXY────────#2^-0.5
│     │           │           │
│     │           │           Z^0.0
│     │           │           │
│     │           @───────────@^V2_3
│     │           │           │
│     │           ×───────────×
│     │           │           │
│     @───────────@^V0_3      │
│     │           │           │
│     ×───────────×           │
│     │           │           │
@─────@^V1_3      @───────────@^V0_2
│     │           │           │
×─────×           ×───────────×
│     │           │           │
Z^0.0 @───────────@^V1_2      │
│     │           │           │
│     ×───────────×           │
│     │           │           │
#2────YXXY^0.5    │           │
│     │           │           │
Z^0.0 Z^0.0       │           │
│     │           │           │
│     #2──────────YXXY^0.608  │
│     │           │           │
#2────YXXY^-0.333 Z^0.0       │
│     │           │           │
Z^0.0 Z^0.0       #2──────────YXXY^0.667
│     │           │           │
│     #2──────────YXXY        Z
│     │           │           │
#2────YXXY^-0.392 Z           Z
│     │           │           │
Z     Z^0.0       Z^U1        │
│     │           │           │
Z^U3  Z^U2        Z           │
│     │           │           │
Z     Z^0.0       │           │
│     │           │           │
#2────YXXY^0.392  │           │
│     │           │           │
Z^0.0 #2──────────YXXY^-1     │
│     │           │           │
│     Z^0.0       #2──────────YXXY^-0.667
│     │           │           │
#2────YXXY^0.333  Z^0.0       │
│     │           │           │
Z^0.0 #2──────────YXXY^-0.608 │
│     │           │           │
│     Z^0.0       │           │
│     │           │           │
#2────YXXY^-0.5   │           │
│     │           │           │
Z^0.0 │           │           │
│     │           │           │
""".strip()


def test_split_operator_trotter_ansatz_default_initial_params_length():

    ansatz = SplitOperatorTrotterAnsatz(hubbard_hamiltonian)
    assert len(ansatz.default_initial_params()) == len(ansatz.params)
