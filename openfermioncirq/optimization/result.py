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

"""Classes for storing the results of running an optimization algorithm."""

from typing import Iterable, List, Optional, TYPE_CHECKING, Tuple

import numpy
import pandas

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from openfermioncirq.optimization.algorithm import OptimizationParams


class OptimizationResult:
    """A result from optimizing a black-box objective function.

    Attributes:
        optimal_value: The best value of the objective function found by the
            optimizer.
        optimal_parameters: The inputs to the objective function which yield the
            optimal value.
        num_evaluations: The number of times the objective function was
            evaluated in the course of the optimization.
        cost_spent: For objective functions with a cost model, the total cost
            spent on function evaluations.
        function_values: A list of tuples storing function values of evaluated
            points. The tuples contain three objects. The first is a function
            value, the second is the cost that was used for the evaluation
            (or None if there was no cost), and the third is the point that
            was evaluated (or None if the black box was initialized with
            `save_x_vals` set to False).
        wait_times: A list of floats. The i-th float float represents the time
            elapsed between the i-th and (i+1)-th times that the black box
            was queried. Time is recorded using ``time.time()``.
        time: The time, in seconds, it took to obtain the result.
        seed: A random number generator seed used to produce the result.
        status: A status flag set by the optimizer.
        message: A message returned by the optimizer.
    """

    def __init__(self,
                 optimal_value: float,
                 optimal_parameters: numpy.ndarray,
                 num_evaluations: Optional[int]=None,
                 cost_spent: Optional[float]=None,
                 function_values: Optional[List[Tuple[
                     float, Optional[float], Optional[numpy.ndarray]
                     ]]]=None,
                 wait_times: Optional[List[float]]=None,
                 time: Optional[int]=None,
                 seed: Optional[int]=None,
                 status: Optional[int]=None,
                 message: Optional[str]=None) -> None:
        self.optimal_value = optimal_value
        self.optimal_parameters = optimal_parameters
        self.num_evaluations = num_evaluations
        self.cost_spent = cost_spent
        self.function_values = function_values
        self.wait_times = wait_times
        self.time = time
        self.seed = seed
        self.status = status
        self.message = message


class OptimizationTrialResult:
    """The results from multiple repetitions of an optimization run.

    Attributes:
        data_frame: A pandas DataFrame storing the results of each repetition
            of the optimization run. It has the following columns:
                optimal_value: The optimal value found.
                optimal_parameters: The function input corresponding to the
                    optimal value.
                num_evaluations: The number of function evaluations used
                    by the optimization algorithm.
                cost_spent: The total cost spent on function evaluations.
                seed: A random number generator seed used by the repetition.
                status: A status returned by the optimization algorithm.
                message: A message returned by the optimization algorithm.
                time: The time it took for the repetition to complete.
                average_wait_time: The average time used by the optimizer to
                    decide on the next evaluation point.
        params: An OptimizationParams object storing the optimization
            parameters used to obtain the results.
        repetitions: The number of times the optimization run was repeated.
        optimal_value: The optimal value over all repetitions of the run.
        optimal_parameters: The function parameters corresponding to the
            optimal value.
    """

    def __init__(self,
                 results: Iterable[OptimizationResult],
                 params: 'OptimizationParams') -> None:
        self.results = list(results)
        self.params = params
        self.data_frame = pandas.DataFrame(
                {'optimal_value': result.optimal_value,
                 'optimal_parameters': result.optimal_parameters,
                 'num_evaluations': result.num_evaluations,
                 'cost_spent': result.cost_spent,
                 'time': result.time,
                 'seed': result.seed,
                 'status': result.status,
                 'message': result.message}
                for result in results)

    @property
    def repetitions(self) -> int:
        return len(self.data_frame)

    @property
    def optimal_value(self) -> float:
        return self.data_frame['optimal_value'].min()

    @property
    def optimal_parameters(self) -> numpy.ndarray:
        return self.data_frame['optimal_parameters'][
                self.data_frame['optimal_value'].idxmin()]

    def extend(self,
               results: Iterable[OptimizationResult]) -> None:
        new_data_frame = pandas.DataFrame(
                {'optimal_value': result.optimal_value,
                 'optimal_parameters': result.optimal_parameters,
                 'num_evaluations': result.num_evaluations,
                 'cost_spent': result.cost_spent,
                 'time': result.time,
                 'seed': result.seed,
                 'status': result.status,
                 'message': result.message}
                for result in results)
        self.data_frame = pandas.concat([self.data_frame, new_data_frame])
        self.results.extend(results)
