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
import openfermion

from openfermioncirq.variational.ansatzes import SwapNetworkTrotterAnsatz


# Construct a Hubbard model Hamiltonian
hubbard_model = openfermion.fermi_hubbard(2, 2, 1., 4.)
hubbard_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(
        hubbard_model)

# Construct an empty Hamiltonian
zero_hamiltonian = openfermion.DiagonalCoulombHamiltonian(
        one_body=numpy.zeros((4, 4)),
        two_body=numpy.zeros((4, 4)))


def test_swap_network_trotter_ansatz_params():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    assert (set(ansatz.params()) ==
            {cirq.Symbol(name) for name in
                {'T_0_2_0', 'T_4_6_0', 'T_1_3_0', 'T_5_7_0',
                 'T_0_4_0', 'T_2_6_0', 'T_1_5_0', 'T_3_7_0',
                 'V_0_1_0', 'V_2_3_0', 'V_4_5_0', 'V_6_7_0'}})

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian, iterations=2)
    assert (set(ansatz.params()) ==
            {cirq.Symbol(name) for name in
                {'T_0_2_0', 'T_4_6_0', 'T_1_3_0', 'T_5_7_0',
                 'T_0_4_0', 'T_2_6_0', 'T_1_5_0', 'T_3_7_0',
                 'V_0_1_0', 'V_2_3_0', 'V_4_5_0', 'V_6_7_0',
                 'T_0_2_1', 'T_4_6_1', 'T_1_3_1', 'T_5_7_1',
                 'T_0_4_1', 'T_2_6_1', 'T_1_5_1', 'T_3_7_1',
                 'V_0_1_1', 'V_2_3_1', 'V_4_5_1', 'V_6_7_1'}})


def test_swap_network_trotter_ansatz_param_bounds():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian)
    assert list(symbol.name for symbol in ansatz.params()) == [
            'V_0_1_0', 'T_0_2_0', 'T_0_4_0', 'T_1_3_0',
            'T_1_5_0', 'V_2_3_0', 'T_2_6_0', 'T_3_7_0',
            'V_4_5_0', 'T_4_6_0', 'T_5_7_0', 'V_6_7_0']
    assert ansatz.param_bounds() == [
            (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0),
            (-2.0, 2.0), (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0),
            (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]


def test_swap_network_trotter_ansatz_circuit():

    complete_ansatz = SwapNetworkTrotterAnsatz(
            zero_hamiltonian,
            include_all_xxyy=True,
            include_all_yxxy=True,
            include_all_cz=True,
            include_all_z=True)
    circuit = complete_ansatz.circuit
    cirq.testing.assert_has_diagram(circuit, """
0       1            2            3
│       │            │            │
XXYY────XXYY^T_0_1_0 XXYY─────────XXYY^T_2_3_0
│       │            │            │
YXXY────#2^W_0_1_0   YXXY─────────#2^W_2_3_0
│       │            │            │
@───────@^V_0_1_0    @────────────@^V_2_3_0
│       │            │            │
×ᶠ──────×ᶠ           ×ᶠ───────────×ᶠ
│       │            │            │
│       XXYY─────────XXYY^T_0_3_0 │
│       │            │            │
│       YXXY─────────#2^W_0_3_0   │
│       │            │            │
│       @────────────@^V_0_3_0    │
│       │            │            │
│       ×ᶠ───────────×ᶠ           │
│       │            │            │
XXYY────XXYY^T_1_3_0 XXYY─────────XXYY^T_0_2_0
│       │            │            │
YXXY────#2^W_1_3_0   YXXY─────────#2^W_0_2_0
│       │            │            │
@───────@^V_1_3_0    @────────────@^V_0_2_0
│       │            │            │
×ᶠ──────×ᶠ           ×ᶠ───────────×ᶠ
│       │            │            │
Z^U_3_0 XXYY─────────XXYY^T_1_2_0 Z^U_0_0
│       │            │            │
│       YXXY─────────#2^W_1_2_0   │
│       │            │            │
│       @────────────@^V_1_2_0    │
│       │            │            │
│       ×ᶠ───────────×ᶠ           │
│       │            │            │
│       Z^U_2_0      Z^U_1_0      │
│       │            │            │
│       @────────────@^V_1_2_0    │
│       │            │            │
│       #2───────────YXXY^W_1_2_0 │
│       │            │            │
│       XXYY─────────XXYY^T_1_2_0 │
│       │            │            │
│       ×ᶠ───────────×ᶠ           │
│       │            │            │
@───────@^V_1_3_0    @────────────@^V_0_2_0
│       │            │            │
#2──────YXXY^W_1_3_0 #2───────────YXXY^W_0_2_0
│       │            │            │
XXYY────XXYY^T_1_3_0 XXYY─────────XXYY^T_0_2_0
│       │            │            │
×ᶠ──────×ᶠ           ×ᶠ───────────×ᶠ
│       │            │            │
│       @────────────@^V_0_3_0    │
│       │            │            │
│       #2───────────YXXY^W_0_3_0 │
│       │            │            │
│       XXYY─────────XXYY^T_0_3_0 │
│       │            │            │
│       ×ᶠ───────────×ᶠ           │
│       │            │            │
@───────@^V_0_1_0    @────────────@^V_2_3_0
│       │            │            │
#2──────YXXY^W_0_1_0 #2───────────YXXY^W_2_3_0
│       │            │            │
XXYY────XXYY^T_0_1_0 XXYY─────────XXYY^T_2_3_0
│       │            │            │
×ᶠ──────×ᶠ           ×ᶠ───────────×ᶠ
│       │            │            │
""", transpose=True)

    hubbard_ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian, iterations=2)
    circuit = hubbard_ansatz.circuit
    cirq.testing.assert_has_diagram(circuit, """
0    1            2            3         4  5         6            7
│    │            │            │         │  │         │            │
@────@^V_0_1_0    @────────────@^V_2_3_0 @──@^V_4_5_0 @────────────@^V_6_7_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           ×ᶠ────────×ᶠ ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_1_3_0 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_4_6_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_1_5_0 │         │  XXYY──────XXYY^T_2_6_0 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_3_7_0 ×ᶠ────────×ᶠ XXYY──────XXYY^T_0_4_0 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_5_7_0 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_0_2_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           ×ᶠ────────×ᶠ ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_5_7_0 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_0_2_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_3_7_0 │         │  XXYY──────XXYY^T_0_4_0 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_1_5_0 ×ᶠ────────×ᶠ XXYY──────XXYY^T_2_6_0 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_1_3_0 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_4_6_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
@────@^V_0_1_0    @────────────@^V_2_3_0 @──@^V_4_5_0 @────────────@^V_6_7_0
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
@────@^V_0_1_1    @────────────@^V_2_3_1 @──@^V_4_5_1 @────────────@^V_6_7_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           ×ᶠ────────×ᶠ ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_1_3_1 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_4_6_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_1_5_1 │         │  XXYY──────XXYY^T_2_6_1 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_3_7_1 ×ᶠ────────×ᶠ XXYY──────XXYY^T_0_4_1 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_5_7_1 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_0_2_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           ×ᶠ────────×ᶠ ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_5_7_1 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_0_2_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_3_7_1 │         │  XXYY──────XXYY^T_0_4_1 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    XXYY─────────XXYY^T_1_5_1 ×ᶠ────────×ᶠ XXYY──────XXYY^T_2_6_1 │
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
XXYY─XXYY^T_1_3_1 ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        XXYY─────────XXYY^T_4_6_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           │            ×ᶠ────────×ᶠ │         ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
│    ×ᶠ───────────×ᶠ           │         │  ×ᶠ────────×ᶠ           │
│    │            │            │         │  │         │            │
@────@^V_0_1_1    @────────────@^V_2_3_1 @──@^V_4_5_1 @────────────@^V_6_7_1
│    │            │            │         │  │         │            │
×ᶠ───×ᶠ           ×ᶠ───────────×ᶠ        ×ᶠ─×ᶠ        ×ᶠ───────────×ᶠ
│    │            │            │         │  │         │            │
""", transpose=True)


def test_swap_network_trotter_ansatz_default_initial_params_length():

    ansatz = SwapNetworkTrotterAnsatz(hubbard_hamiltonian,
                                      include_all_yxxy=True,
                                      include_all_z=True)
    assert len(ansatz.default_initial_params()) == len(list(ansatz.params()))
