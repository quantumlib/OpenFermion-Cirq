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

"""Defines the interface for black box optimization algorithms."""

from typing import Any, Optional

import abc

import numpy

from openfermioncirq.optimization.black_box import BlackBox
from openfermioncirq.optimization.result import OptimizationResult


class OptimizationAlgorithm(metaclass=abc.ABCMeta):
    """An optimization algorithm for black-box objective functions.

    We use the convention that the optimization algorithm should try to minimize
    (rather than maximize) the value of the objective function.

    In order to work with some routines that save optimization results,
    instances of this class must be picklable. See
    https://docs.python.org/3/library/pickle.html
    for details.

    Attributes:
        options: Options for the algorithm.
    """

    def __init__(self, options: Optional[Any]=None) -> None:
        """
        Args:
            options: Options for the algorithm.
        """
        # TODO options is probably always a Dict
        self.options = options or self.default_options()

    def default_options(self):
        """Default options for the algorithm."""
        # TODO tailor default options to the particular problem (e.g. number of
        #      parameters)
        return {}

    @abc.abstractmethod
    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        """Perform the optimization and return the result.

        Args:
            black_box: A BlackBox encapsulating the objective function.
            initial_guess: An initial point at which to evaluate the objective
                function.
            initial_guess_array: An array of initial points at which to evaluate
                the objective function, for algorithms that can use multiple
                initial points. This is a 2d numpy array with each row
                representing one initial point.
        """
        pass

    @property
    def name(self) -> str:
        """A name for the optimization algorithm."""
        return type(self).__name__


class OptimizationParams:
    """Parameters for an optimization run.

    Attributes:
        algorithm: The algorithm to use.
        initial_guess: An initial guess for the algorithm to use.
        initial_guess_array: An array of initial guesses for the algorithm
            to use. This is a 2d numpy array with each row representing
            one initial point.
        cost_of_evaluate: A cost value associated with the `evaluate`
            method of the BlackBox to be optimized. For use with black boxes
            with a noise and cost model.
    """

    def __init__(self,
                 algorithm: OptimizationAlgorithm,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None,
                 cost_of_evaluate: Optional[float]=None) -> None:
        """Construct a parameters object by setting its attributes."""
        self.algorithm = algorithm
        self.initial_guess = initial_guess
        self.initial_guess_array = initial_guess_array
        self.cost_of_evaluate = cost_of_evaluate
