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


from openfermioncirq.variational.ansatzes import SwapNetworkTrotterHubbardAnsatz


def test_swap_network_trotter_hubbard_ansatz_param_bounds():
    ansatz = SwapNetworkTrotterHubbardAnsatz(3, 1, 1.0, 4.0, periodic=False)
    assert list(symbol.name for symbol in ansatz.params()) == [
            'Th_0', 'V_0',]
    assert ansatz.param_bounds() == [
            (-2.0, 2.0), (-1.0, 1.0)]

    ansatz = SwapNetworkTrotterHubbardAnsatz(1, 4, 1.0, 4.0, periodic=False)
    assert list(symbol.name for symbol in ansatz.params()) == [
            'Tv_0', 'V_0',]
    assert ansatz.param_bounds() == [
            (-2.0, 2.0), (-1.0, 1.0)]

    ansatz = SwapNetworkTrotterHubbardAnsatz(3, 2, 1.0, 4.0)
    assert list(symbol.name for symbol in ansatz.params()) == [
            'Th_0', 'Tv_0', 'V_0',]
    assert ansatz.param_bounds() == [
            (-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]
