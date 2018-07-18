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

from typing import Iterable, Optional, TYPE_CHECKING, Union

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
        seed: A random number generator seed used to produce the result.
        status: A status flag set by the optimizer.
        message: A message returned by the optimizer.
    """

    def __init__(self,
                 optimal_value: float,
                 optimal_parameters: numpy.ndarray,
                 num_evaluations: int,
                 cost_spent: Optional[float]=None,
                 seed: Optional[int]=None,
                 status: Optional[int]=None,
                 message: Optional[str]=None) -> None:
        self.optimal_value = optimal_value
        self.optimal_parameters = optimal_parameters
        self.num_evaluations = num_evaluations
        self.cost_spent = cost_spent
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
        self.data_frame = pandas.DataFrame(
                {'optimal_value': result.optimal_value,
                 'optimal_parameters': result.optimal_parameters,
                 'num_evaluations': result.num_evaluations,
                 'cost_spent': result.cost_spent,
                 'seed': result.seed,
                 'status': result.status,
                 'message': result.message}
                for result in results)
        self.params = params

    @property
    def repetitions(self):
        return len(self.data_frame)

    @property
    def optimal_value(self):
        return self.data_frame['optimal_value'].min()

    @property
    def optimal_parameters(self):
        return self.data_frame['optimal_parameters'][
                self.data_frame['optimal_value'].idxmin()]

    def optimal_value_quantile(self,
                               q: Union[float, numpy.ndarray]=0.5,
                               interpolation='linear'):
        """Return the optimal value at the given quantile.

        This behaves like numpy.percentile.
        """
        return self.data_frame['optimal_value'].quantile(
                q, interpolation=interpolation)

    def num_evaluations_quantile(self,
                                 q: Union[float, numpy.ndarray]=0.5,
                                 interpolation='linear'):
        """Return the number of evaluations used at the given quantile.

        This behaves like numpy.percentile.
        """
        return self.data_frame['num_evaluations'].quantile(
                q, interpolation=interpolation)

    def cost_spent_quantile(self,
                            q: Union[float, numpy.ndarray]=0.5,
                            interpolation='linear'):
        """Return the cost spent at the given quantile.

        This behaves like numpy.percentile.
        """
        return self.data_frame['cost_spent'].quantile(
                q, interpolation=interpolation)
