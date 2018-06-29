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

from typing import Optional

import numpy


class OptimizationResult:
    """The results from optimizing a black-box objective function.

    Attributes:
        optimal_value: The best value of the objective function found by the
            optimizer.
        optimal_parameters: The inputs to the objective function which yield the
            optimal value.
        num_evaluations: The number of times the objective function was
            evaluated in the course of the optimization.
        cost_spent: For objective functions with a cost model, the total cost
            spent on function evaluations.
        initial_guess: The initial guess, if any, used by the optimizer.
        status: A status flag set by the optimizer.
        message: A message returned by the optimizer.
    """

    def __init__(self,
                 optimal_value: float,
                 optimal_parameters: numpy.ndarray,
                 num_evaluations: int,
                 cost_spent: float=0.0,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None,
                 seed: Optional[int]=None,
                 status: Optional[int]=None,
                 message: Optional[str]=None) -> None:
        self.optimal_value = optimal_value
        self.optimal_parameters = optimal_parameters
        self.num_evaluations = num_evaluations
        self.cost_spent = cost_spent
        self.initial_guess = initial_guess
        self.initial_guess_array = initial_guess_array
        self.seed = seed
        self.status = status
        self.message = message
