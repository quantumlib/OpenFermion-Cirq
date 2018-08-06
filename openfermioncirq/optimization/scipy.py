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

"""A wrapper around the local optimization routines implemented in Scipy."""

from typing import Dict, Optional

import numpy
import scipy.optimize

from openfermioncirq.optimization import (BlackBox,
                                          OptimizationResult,
                                          OptimizationAlgorithm)


class ScipyOptimizationAlgorithm(OptimizationAlgorithm):
    """An optimization algorithm from the scipy.optimize module."""

    def __init__(self,
                 options: Optional[Dict]=None,
                 kwargs: Optional[Dict]=None,
                 uses_bounds: bool=True) -> None:
        """
        Args:
            options: The `options` dictionary passed to scipy.optimize.minimize.
            kwargs: Other keyword arguments passed to scipy.optimize.minimize.
                This should NOT include the `bounds` or `options` keyword
                arguments.
            uses_bounds: Whether the algorithm uses bounds on the input
                variables. Set this to False to prevent scipy.optimize.minimize
                from raising a warning if the chosen method does not use bounds.
        """
        self.kwargs = kwargs or {}
        self.uses_bounds = uses_bounds
        super().__init__(options)

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        if initial_guess is None:
            raise ValueError('The chosen optimization algorithm requires an '
                             'initial guess.')
        bounds = black_box.bounds if self.uses_bounds else None
        result = scipy.optimize.minimize(black_box.evaluate,
                                         initial_guess,
                                         bounds=bounds,
                                         options=self.options,
                                         **self.kwargs)
        return OptimizationResult(optimal_value=result.fun,
                                  optimal_parameters=result.x,
                                  num_evaluations=result.nfev,
                                  status=result.status,
                                  message=result.message)

    @property
    def name(self) -> str:
        return self.kwargs.get('method', 'ScipyOptimizationAlgorithm')


COBYLA = ScipyOptimizationAlgorithm(
        kwargs={'method': 'COBYLA'},
        uses_bounds=False)

L_BFGS_B = ScipyOptimizationAlgorithm(
        kwargs={'method': 'L-BFGS-B'})

NELDER_MEAD = ScipyOptimizationAlgorithm(
        kwargs={'method': 'Nelder-Mead'},
        uses_bounds=False)

SLSQP = ScipyOptimizationAlgorithm(
        kwargs={'method': 'SLSQP'})
