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

from openfermioncirq.gates import (
    CXXYY,
    CYXXY,
    CombinedDoubleExcitationGate,
    ControlledXXYYGate,
    ControlledYXXYGate,
    DoubleExcitation,
    DoubleExcitationGate,
    FSWAP,
    FermionicSwapGate,
    rot11,
    rot111,
    XXYY,
    XXYYGate,
    YXXY,
    YXXYGate,
    ZZ,
    ZZGate,
)

from openfermioncirq.primitives import (
    prepare_gaussian_state,
    prepare_slater_determinant)

from openfermioncirq.primitives.bogoliubov_transform import bogoliubov_transform

from openfermioncirq.primitives.swap_network import swap_network

from openfermioncirq.trotter import simulate_trotter

from openfermioncirq.variational import (
    HamiltonianObjective,
    LowRankTrotterAnsatz,
    SplitOperatorTrotterAnsatz,
    SwapNetworkTrotterAnsatz,
    VariationalAnsatz,
    VariationalObjective,
    VariationalStudy)

# Import modules last to avoid circular dependencies
from openfermioncirq import (
    gates,
    optimization,
    primitives,
    trotter,
    variational,
    testing,
)

from ._version import __version__
