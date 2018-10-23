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

"""Defines the interface for a black box objective function."""

from typing import Optional, Sequence, TYPE_CHECKING, Tuple

import abc
import time

import numpy

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


class BlackBox(metaclass=abc.ABCMeta):
    """A black box objective function.

    This class encapsulates the evaluation of an objective function. The
    objective function may be noisy. It is assumed that the objective function
    takes as input an array of real numbers. The dimension of the black box is
    defined to be the length of the input array.

    One can optionally provide a version of the objective function that takes a
    `cost` parameter. This is used to model situations in which the objective
    function is noisy but the amount of noise present can be controlled to some
    extent, i.e., providing a higher cost should reduce the magnitude of the
    noise.

    Attributes:
        dimension: The dimension of the array accepted by the objective
            function.
        bounds: Optional bounds on the inputs to the objective function. This is
            a list of tuples of the form (low, high), where low and high are
            lower and upper bounds on a parameter. The number of tuples
            should be equal to the dimension of the black box.
        cost_of_evaluate: If specified, then calls to `evaluate` will be
            redirected to `evaluate_with_cost` with the specified cost.
    """

    def __init__(self,
                 cost_of_evaluate: Optional[float]=None,
                 **kwargs) -> None:
        """
        Args:
            cost_of_evaluate: An optional cost associated with the
                `evaluate` method. If specified, the `evaluate` method
                will defer to `evaluate_with_cost` with the specified cost.
        """
        self.cost_of_evaluate = cost_of_evaluate

    @abc.abstractproperty
    def dimension(self) -> int:
        """The dimension of the array accepted by the objective function."""
        pass

    @property
    def bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        """Optional bounds on the inputs to the objective function.

        A list of tuples of the form (low, high), where low and high are
        lower and upper bounds on a parameter. The number of tuples
        should be equal to the dimension of the black box.
        """
        return None

    @abc.abstractmethod
    def _evaluate(self,
                  x: numpy.ndarray) -> float:
        """Evaluate the objective function.

        Implement this method when defining a BlackBox.
        """
        pass

    def _evaluate_with_cost(self,
                            x: numpy.ndarray,
                            cost: float) -> float:
        """Evaluate the objective function with a specified cost.

        Implement this method when defining a BlackBox with a cost model.
        """
        # Default: defer to `_evaluate`
        return self._evaluate(x)

    def evaluate(self,
                 x: numpy.ndarray) -> float:
        """Evaluate the objective function."""
        if self.cost_of_evaluate is not None:
            return self.evaluate_with_cost(x, self.cost_of_evaluate)
        return self._evaluate(x)

    def evaluate_with_cost(self,
                           x: numpy.ndarray,
                           cost: float) -> float:
        """Evaluate the objective function with a specified cost.

        This is used to model situations in which it is possible to reduce the
        magnitude of the noise at some cost.
        """
        return self._evaluate_with_cost(x, cost)

    def noise_bounds(self,
                     cost: float,
                     confidence: Optional[float]=None
                     ) -> Tuple[float, float]:
        """Exact or approximate bounds on noise in the objective function.

        Returns a tuple (a, b) such that when `evaluate_with_cost` is called
        with the given cost and returns an approximate function value y, the
        true function value lies in the interval [y + a, y + b]. Thus, it should
        be the case that a <= 0 <= b.

        This function takes an optional `confidence` parameter which is a real
        number strictly between 0 and 1 that gives the confidence level in the
        bound. This is used for situations in which exact bounds on the noise
        cannot be guaranteed. The value can be interpreted as the probability
        that a repeated call to `evaluate_with_cost` with the same cost will
        return a value within the bounds.
        """
        return -numpy.inf, numpy.inf


class StatefulBlackBox(BlackBox):
    """A black box function with memory of evaluations.

    This black box keeps track of the the points that have been evaluated,
    the total cost spent on evaluations, and the time elapsed between
    queries.

    Attributes:
        num_evaluations: The number of times the objective function has been
            evaluated, including noisy evaluations.
        cost_spent: The total cost that has been spent on function evaluations.
        cost_of_evaluate: An optional cost associated with the
            ``evaluate`` method.
        function_values: A list of tuples storing function values of evaluated
            points. The tuples contain three objects. The first is a function
            value, the second is the cost that was used for the evaluation
            (or None if there was no cost), and the third is the point that
            was evaluated (or None if the black box was initialized with
            ``save_x_vals`` set to False.
        wait_times: A list of floats. The i-th float float represents the time
            elapsed between the i-th and (i+1)-th times that the black box
            was queried. Time is recorded using ``time.time()``.
    """

    def __init__(self,
                 save_x_vals: bool=False,
                 **kwargs) -> None:
        """
        Args:
            save_x_vals: Whether to save all points (x values) that the
                black box was queried at. Setting this to True will cause the
                black box to consume a lot more memory. This does not affect
                whether the function values (y values) are saved (they are
                saved no matter what).
        """
        self.function_values = [] \
            # type: List[Tuple[float, Optional[float], Optional[numpy.ndarray]]]
        self.cost_spent = 0.0
        self.wait_times = []  # type: List[float]
        self._save_x_vals = save_x_vals
        self._time_of_last_query = None  # type: Optional[float]
        super().__init__(**kwargs)

    @property
    def num_evaluations(self) -> float:
        """The number of times the objective function has been evaluated."""
        return len(self.function_values)

    def evaluate(self,
                 x: numpy.ndarray) -> float:
        """Evaluate the objective function and update state."""
        # If cost_of_evaluate is set, defer to evaluate_with_cost
        if self.cost_of_evaluate is not None:
            return self.evaluate_with_cost(x, self.cost_of_evaluate)

        if self._time_of_last_query is not None:
            self.wait_times.append(time.time() - self._time_of_last_query)

        val = self._evaluate(x)
        self.function_values.append(
                (val, None, x if self._save_x_vals else None)
        )
        self._time_of_last_query = time.time()
        return val

    def evaluate_with_cost(self,
                           x: numpy.ndarray,
                           cost: float) -> float:
        """Evaluate the objective function with a cost and update state."""
        if self._time_of_last_query is not None:
            self.wait_times.append(time.time() - self._time_of_last_query)

        val = self._evaluate_with_cost(x, cost)
        self.function_values.append(
                (val, cost, x if self._save_x_vals else None)
        )
        self.cost_spent += cost
        self._time_of_last_query = time.time()
        return val
