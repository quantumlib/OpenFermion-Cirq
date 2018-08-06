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

"""Optimization algorithms and related classes."""

from openfermioncirq.optimization.algorithm import (
    OptimizationAlgorithm,
    OptimizationParams)

from openfermioncirq.optimization.black_box import (
    BlackBox,
    StatefulBlackBox)

from openfermioncirq.optimization.result import (
    OptimizationResult,
    OptimizationTrialResult)

from openfermioncirq.optimization.scipy import (
    COBYLA,
    L_BFGS_B,
    NELDER_MEAD,
    SLSQP,
    ScipyOptimizationAlgorithm)
