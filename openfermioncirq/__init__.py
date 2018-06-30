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

from openfermioncirq.ops import (
        CCZ,
        CXXYY,
        CYXXY,
        ControlledXXYYGate,
        ControlledYXXYGate,
        FSWAP,
        FermionicSwapGate,
        ISWAP,
        PlanarQubit,
        Rot111Gate,
        XXYY,
        XXYYGate,
        YXXY,
        YXXYGate,
        ZZ,
        ZZGate)

from openfermioncirq.state_preparation import (
        bogoliubov_transform,
        prepare_gaussian_state,
        prepare_slater_determinant)

from openfermioncirq.swap_network import swap_network

from openfermioncirq.trotter import simulate_trotter
from openfermioncirq.variational import (
        SwapNetworkTrotterAnsatz,
        VariationalAnsatz,
        VariationalStudy)

from openfermioncirq.variational import (
        HamiltonianVariationalStudy,
        SwapNetworkTrotterAnsatz,
        VariationalAnsatz,
        VariationalStudy)

from ._version import __version__
