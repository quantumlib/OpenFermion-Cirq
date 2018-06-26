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

from typing import Optional, Sequence, Tuple

import numpy

from cirq import abc


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
    """

    @property  # type: ignore
    @abc.abstractmethod
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
    def evaluate(self,
                 x: numpy.ndarray) -> float:
        """Evaluate the objective function."""
        pass

    def evaluate_with_cost(self,
                           x: numpy.ndarray,
                           cost: float) -> float:
        """Evaluate the objective function with a specified cost.

        This is used to model situations in which it is possible to reduce the
        magnitude of the noise at some cost.
        """
        # Default: defer to `evaluate`
        return self.evaluate(x)
