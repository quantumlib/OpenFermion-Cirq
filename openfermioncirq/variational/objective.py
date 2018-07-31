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

from typing import Optional, Tuple, Union

import numpy

import cirq
from cirq import abc


class VariationalObjective(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def value(self,
              trial_result: Union[cirq.TrialResult,
                                  cirq.google.XmonSimulateTrialResult]
              ) -> float:
        """The evaluation function for a circuit output.

        A variational quantum algorithm will attempt to minimize this value over
        possible settings of the parameters.
        """
        pass

    def noise(self, cost: Optional[float]=None) -> float:
        """Artificial noise that may be added to the true objective value.

        The `cost` argument is used to model situations in which it is possible
        to reduce the magnitude of the noise at some cost.
        """
        # Default: no noise
        return 0.0

    def noise_bounds(self,
                     cost: float,
                     confidence: Optional[float]=None
                     ) -> Tuple[float, float]:
        """Exact or approximate bounds on noise.

        Returns a tuple (a, b) such that when `noise` is called with the given
        cost, the returned value lies between a and b. It should be the case
        that a <= 0 <= b.

        This function takes an optional `confidence` parameter which is a real
        number strictly between 0 and 1 that gives the probability of the bounds
        being correct. This is used for situations in which exact bounds on the
        noise cannot be guaranteed.
        """
        return -numpy.inf, numpy.inf
