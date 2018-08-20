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

import cirq
import openfermion

from openfermioncirq.variational.ansatzes import LowRankTrotterAnsatz


def load_molecular_hamiltonian(
        geometry,
        basis,
        multiplicity,
        description,
        n_active_electrons,
        n_active_orbitals):

    molecule = openfermion.MolecularData(
            geometry, basis, multiplicity, description=description)
    molecule.load()

    n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
    occupied_indices = list(range(n_core_orbitals))
    active_indices = list(range(n_core_orbitals,
                                n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)


# 4-qubit H2 2-2 with bond length 0.7414
bond_length = 0.7414
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
h2_hamiltonian = load_molecular_hamiltonian(
        geometry, 'sto-3g', 1, format(bond_length), 2, 2)


def test_low_rank_trotter_ansatz_params():

    ansatz = LowRankTrotterAnsatz(h2_hamiltonian, final_rank=2)
    assert (set(ansatz.params()) ==
            {cirq.Symbol(name) for name in
                {'U_0_0', 'U_1_0', 'U_2_0', 'U_3_0',
                 'U_0_0_0', 'U_1_0_0', 'U_2_0_0', 'U_3_0_0',
                 'U_0_1_0', 'U_1_1_0', 'U_2_1_0', 'U_3_1_0',
                 'V_0_1_0_0', 'V_0_2_0_0', 'V_0_3_0_0',
                 'V_1_2_0_0', 'V_1_3_0_0', 'V_2_3_0_0',
                 'V_0_1_1_0', 'V_0_2_1_0', 'V_0_3_1_0',
                 'V_1_2_1_0', 'V_1_3_1_0', 'V_2_3_1_0'}})


def test_low_rank_trotter_ansatz_param_bounds():

    ansatz = LowRankTrotterAnsatz(h2_hamiltonian, final_rank=2)
    assert ansatz.param_bounds() == [(-1.0, 1.0)] * len(list(ansatz.params()))

    ansatz = LowRankTrotterAnsatz(
            h2_hamiltonian,
            final_rank=2,
            include_all_xxyy=True,
            include_all_cz=True,
            include_all_z=True)
    assert list(symbol.name for symbol in ansatz.params()) == [
            'U_0_0', 'U_0_0_0', 'U_0_1_0', 'U_1_0',
            'U_1_0_0', 'U_1_1_0', 'U_2_0', 'U_2_0_0',
            'U_2_1_0', 'U_3_0', 'U_3_0_0', 'U_3_1_0',
            'T_0_1_0', 'V_0_1_0_0', 'V_0_1_1_0', 'T_0_2_0',
            'V_0_2_0_0', 'V_0_2_1_0', 'T_0_3_0', 'V_0_3_0_0',
            'V_0_3_1_0', 'T_1_2_0', 'V_1_2_0_0', 'V_1_2_1_0',
            'T_1_3_0', 'V_1_3_0_0', 'V_1_3_1_0', 'T_2_3_0',
            'V_2_3_0_0', 'V_2_3_1_0']
    assert ansatz.param_bounds() == [
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0),
            (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0),
            (-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0),
            (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0),
            (-1.0, 1.0), (-1.0, 1.0)]

def test_low_rank_trotter_ansatz_circuit():

    ansatz = LowRankTrotterAnsatz(
            h2_hamiltonian, final_rank=2, omit_final_swaps=True)
    circuit = ansatz.circuit
    cirq.DropNegligible().optimize_circuit(circuit)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0         1           2           3
│         │           │           │
×ᶠ────────×ᶠ          ×ᶠ──────────×ᶠ
│         │           │           │
│         ×ᶠ──────────×ᶠ          │
│         │           │           │
×ᶠ────────×ᶠ          ×ᶠ──────────×ᶠ
│         │           │           │
Z^U_3_0   ×ᶠ──────────×ᶠ          Z^U_0_0
│         │           │           │
│         Z^U_2_0     Z^U_1_0     │
│         │           │           │
│         #2──────────YXXY^-1     │
│         │           │           │
│         │           #2──────────YXXY^-1
│         │           │           │
#2────────YXXY^-1     │           │
│         │           │           │
│         #2──────────YXXY^-1     │
│         │           │           │
│         │           @───────────@^V_0_1_0_0
│         │           │           │
@─────────@^V_2_3_0_0 ×───────────×
│         │           │           │
×─────────×           │           │
│         │           │           │
│         @───────────@^V_0_3_0_0 │
│         │           │           │
│         ×───────────×           │
│         │           │           │
@─────────@^V_0_2_0_0 @───────────@^V_1_3_0_0
│         │           │           │
×─────────×           ×───────────×
│         │           │           │
Z^U_0_0_0 @───────────@^V_1_2_0_0 Z^U_3_0_0
│         │           │           │
Z         ×───────────×           Z
│         │           │           │
│         Z^U_1_0_0   Z^U_2_0_0   │
│         │           │           │
│         YXXY────────#2^-1       │
│         │           │           │
YXXY──────#2^0.5      │           │
│         │           │           │
│         │           YXXY────────#2^-0.5
│         │           │           │
│         YXXY────────#2^-1       │
│         │           │           │
@─────────@^V_0_1_1_0 │           │
│         │           │           │
×─────────×           @───────────@^V_2_3_1_0
│         │           │           │
│         │           ×───────────×
│         │           │           │
│         @───────────@^V_0_3_1_0 │
│         │           │           │
│         ×───────────×           │
│         │           │           │
@─────────@^V_1_3_1_0 @───────────@^V_0_2_1_0
│         │           │           │
×─────────×           ×───────────×
│         │           │           │
Z^U_3_1_0 @───────────@^V_1_2_1_0 Z^U_0_1_0
│         │           │           │
│         ×───────────×           Z
│         │           │           │
│         Z^U_2_1_0   Z^U_1_1_0   │
│         │           │           │
│         Z           │           │
│         │           │           │
│         #2──────────YXXY^-1     │
│         │           │           │
│         │           #2──────────YXXY^-0.5
│         │           │           │
#2────────YXXY^0.5    │           │
│         │           │           │
│         #2──────────YXXY^-1     │
│         │           │           │
""".strip()

    ansatz = LowRankTrotterAnsatz(
            h2_hamiltonian,
            final_rank=2,
            include_all_xxyy=True,
            include_all_cz=True,
            include_all_z=True,
            iterations=2)
    circuit = ansatz.circuit
    cirq.DropNegligible().optimize_circuit(circuit)
    assert circuit.to_text_diagram(transpose=True).strip() == """
0         1            2            3
│         │            │            │
XXYY──────XXYY^T_0_1_0 XXYY─────────XXYY^T_2_3_0
│         │            │            │
×ᶠ────────×ᶠ           ×ᶠ───────────×ᶠ
│         │            │            │
│         XXYY─────────XXYY^T_0_3_0 │
│         │            │            │
│         ×ᶠ───────────×ᶠ           │
│         │            │            │
XXYY──────XXYY^T_1_3_0 XXYY─────────XXYY^T_0_2_0
│         │            │            │
×ᶠ────────×ᶠ           ×ᶠ───────────×ᶠ
│         │            │            │
Z^U_3_0   XXYY─────────XXYY^T_1_2_0 Z^U_0_0
│         │            │            │
│         ×ᶠ───────────×ᶠ           │
│         │            │            │
│         Z^U_2_0      Z^U_1_0      │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            #2───────────YXXY^-1
│         │            │            │
#2────────YXXY^-1      │            │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            @────────────@^V_0_1_0_0
│         │            │            │
@─────────@^V_2_3_0_0  ×────────────×
│         │            │            │
×─────────×            │            │
│         │            │            │
│         @────────────@^V_0_3_0_0  │
│         │            │            │
│         ×────────────×            │
│         │            │            │
@─────────@^V_0_2_0_0  @────────────@^V_1_3_0_0
│         │            │            │
×─────────×            ×────────────×
│         │            │            │
Z^U_0_0_0 @────────────@^V_1_2_0_0  Z^U_3_0_0
│         │            │            │
Z         ×────────────×            Z
│         │            │            │
│         Z^U_1_0_0    Z^U_2_0_0    │
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
YXXY──────#2^0.5       │            │
│         │            │            │
│         │            YXXY─────────#2^-0.5
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
@─────────@^V_0_1_1_0  │            │
│         │            │            │
×─────────×            @────────────@^V_2_3_1_0
│         │            │            │
│         │            ×────────────×
│         │            │            │
│         @────────────@^V_0_3_1_0  │
│         │            │            │
│         ×────────────×            │
│         │            │            │
@─────────@^V_1_3_1_0  @────────────@^V_0_2_1_0
│         │            │            │
×─────────×            ×────────────×
│         │            │            │
Z^U_3_1_0 @────────────@^V_1_2_1_0  Z^U_0_1_0
│         │            │            │
│         ×────────────×            Z
│         │            │            │
│         Z^U_2_1_0    Z^U_1_1_0    │
│         │            │            │
│         Z            │            │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            #2───────────YXXY^-0.5
│         │            │            │
#2────────YXXY^0.5     │            │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            XXYY─────────XXYY^T_0_1_1
│         │            │            │
XXYY──────XXYY^T_2_3_1 ×ᶠ───────────×ᶠ
│         │            │            │
×ᶠ────────×ᶠ           │            │
│         │            │            │
│         XXYY─────────XXYY^T_0_3_1 │
│         │            │            │
│         ×ᶠ───────────×ᶠ           │
│         │            │            │
XXYY──────XXYY^T_0_2_1 XXYY─────────XXYY^T_1_3_1
│         │            │            │
×ᶠ────────×ᶠ           ×ᶠ───────────×ᶠ
│         │            │            │
Z^U_0_1   XXYY─────────XXYY^T_1_2_1 Z^U_3_1
│         │            │            │
│         ×ᶠ───────────×ᶠ           │
│         │            │            │
│         Z^U_1_1      Z^U_2_1      │
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
YXXY──────#2^-1        │            │
│         │            │            │
│         │            YXXY─────────#2^-1
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
@─────────@^V_0_1_0_1  │            │
│         │            │            │
×─────────×            @────────────@^V_2_3_0_1
│         │            │            │
│         │            ×────────────×
│         │            │            │
│         @────────────@^V_0_3_0_1  │
│         │            │            │
│         ×────────────×            │
│         │            │            │
@─────────@^V_1_3_0_1  @────────────@^V_0_2_0_1
│         │            │            │
×─────────×            ×────────────×
│         │            │            │
Z^U_3_0_1 @────────────@^V_1_2_0_1  Z^U_0_0_1
│         │            │            │
Z         ×────────────×            Z
│         │            │            │
│         Z^U_2_0_1    Z^U_1_0_1    │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            #2───────────YXXY^0.5
│         │            │            │
#2────────YXXY^-0.5    │            │
│         │            │            │
│         #2───────────YXXY^-1      │
│         │            │            │
│         │            @────────────@^V_0_1_1_1
│         │            │            │
@─────────@^V_2_3_1_1  ×────────────×
│         │            │            │
×─────────×            │            │
│         │            │            │
│         @────────────@^V_0_3_1_1  │
│         │            │            │
│         ×────────────×            │
│         │            │            │
@─────────@^V_0_2_1_1  @────────────@^V_1_3_1_1
│         │            │            │
×─────────×            ×────────────×
│         │            │            │
Z^U_0_1_1 @────────────@^V_1_2_1_1  Z^U_3_1_1
│         │            │            │
Z         ×────────────×            │
│         │            │            │
│         Z^U_1_1_1    Z^U_2_1_1    │
│         │            │            │
│         │            Z            │
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
YXXY──────#2^-0.5      │            │
│         │            │            │
│         │            YXXY─────────#2^0.5
│         │            │            │
│         YXXY─────────#2^-1        │
│         │            │            │
""".strip()


def test_swap_network_trotter_ansatz_default_initial_params_length():

    ansatz = LowRankTrotterAnsatz(h2_hamiltonian)
    assert len(ansatz.default_initial_params()) == len(list(ansatz.params()))
