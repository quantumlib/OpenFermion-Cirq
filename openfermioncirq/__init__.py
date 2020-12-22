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

# pylint: disable=wrong-import-position

import warnings

warnings.warn(
    'OpenFermion-Cirq is deprecated and no longer maintained. '
    'Its functionality has been merged into OpenFermion. '
    'To uninstall OpenFermion-Cirq and upgrade to the latest version of '
    'OpenFermion using pip, execute '
    '`pip uninstall openfermioncirq` followed by '
    '`pip install --upgrade openfermion`.',
    DeprecationWarning,
    stacklevel=2)

from openfermioncirq.gates import (
    CRxxyy,
    CRyxxy,
    CXXYYPowGate,
    CYXXYPowGate,
    DoubleExcitation,
    DoubleExcitationGate,
    FSWAP,
    FSwapPowGate,
    Rxxyy,
    Ryxxy,
    Rzz,
    rot11,
    rot111,
    XXYYPowGate,
    YXXYPowGate,
    fermionic_simulation_gates_from_interaction_operator,
    ParityPreservingFermionicGate,
    QuadraticFermionicSimulationGate,
    CubicFermionicSimulationGate,
    QuarticFermionicSimulationGate,
)

from openfermioncirq.primitives import (
    ffft,
    prepare_gaussian_state,
    prepare_slater_determinant,
)

from openfermioncirq.primitives.bogoliubov_transform import bogoliubov_transform

from openfermioncirq.primitives.swap_network import swap_network

from openfermioncirq.trotter import simulate_trotter

from openfermioncirq.variational import (
    HamiltonianObjective,
    LowRankTrotterAnsatz,
    SplitOperatorTrotterAnsatz,
    SwapNetworkTrotterAnsatz,
    SwapNetworkTrotterHubbardAnsatz,
    VariationalAnsatz,
    VariationalObjective,
    VariationalStudy,
)

# Import modules last to avoid circular dependencies
from openfermioncirq import (
    gates,
    optimization,
    primitives,
    trotter,
    variational,
    testing,
)

from openfermioncirq._version import __version__

# Deprecated
# pylint: disable=wrong-import-order
import sys as _sys
import warnings as _warnings
from openfermioncirq._compat import wrap_module as _wrap_module
with _warnings.catch_warnings():
    _warnings.simplefilter('ignore')
    from openfermioncirq.gates.common_gates import (
        XXYY,
        YXXY,
    )
    from openfermioncirq.gates.three_qubit_gates import (
        CXXYY,
        CYXXY,
    )
_deprecated_constants = {
    'XXYY': ('v0.5.0', 'Use cirq.ISWAP with negated exponent, instead'),
    'YXXY': ('v0.5.0', 'Use cirq.PhasedISwapPowGate, instead.'),
    'CXXYY': ('v0.5.0', 'Use cirq.ControlledGate and cirq.ISWAP with '
              'negated exponent, instead'),
    'CYXXY': ('v0.5.0', 'Use cirq.ControlledGate and '
              'cirq.PhasedISwapPowGate, instead.'),
}
_sys.modules[__name__] = _wrap_module(_sys.modules[__name__],
                                      _deprecated_constants)
