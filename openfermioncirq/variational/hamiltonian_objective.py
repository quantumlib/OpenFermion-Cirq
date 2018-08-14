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

"""A class for studying variational ansatzes with an associated Hamiltonian."""

from typing import Optional, Tuple, Union

import numpy
import scipy.special

import cirq
import openfermion

from openfermioncirq.variational.objective import VariationalObjective


class HamiltonianObjective(VariationalObjective):
    """A variational objective associated with a Hamiltonian.

    The value of the objective is the expectation value of the Hamiltonian.
    The noise and cost model assume that one estimates the expectation value
    by measuring individual Pauli terms of the Hamiltonian a finite number
    of times. The resulting estimate is modeled as a random value drawn from
    a normal distribution centered at the true expectation value and whose
    variance is inversely proportional to the number of measurements taken.
    The cost corresponds to the number of measurements performed.

    Attributes:
        hamiltonian: The Hamiltonian of interest, represented
            as a FermionOperator, QubitOperator, InteractionOperator, or
            DiagonalCoulombHamiltonian. If the input is not a QubitOperator,
            then the Jordan-Wigner transform is used.
        variance_bound: The squared one-norm of the coefficients of
            the Pauli terms in the Jordan-Wigner transformed Hamiltonian.
            This gives an estimate of the variance of an energy measurement
            with a certain measurement strategy; see arXiv:1801.03524 for
            a derivation.
    """

    def __init__(self,
                 hamiltonian: Union[
                     openfermion.DiagonalCoulombHamiltonian,
                     openfermion.FermionOperator,
                     openfermion.InteractionOperator,
                     openfermion.QubitOperator],
                 use_linear_op: bool=False) -> None:
        """
        Args:
            hamiltonian: The Hamiltonian.
            use_linear_op: Whether to use a LinearOperator instead of a sparse
                matrix to compute expectation values. Using a LinearOperator
                is more memory-efficient but results in much slower expectation
                value computation.
        """
        self.hamiltonian = hamiltonian

        if isinstance(hamiltonian, openfermion.QubitOperator):
            hamiltonian_qubit_op = hamiltonian
        else:
            hamiltonian_qubit_op = openfermion.jordan_wigner(hamiltonian)

        if use_linear_op:
            self._hamiltonian_linear_op = openfermion.LinearQubitOperator(
                    hamiltonian_qubit_op)
        else:
            self._hamiltonian_linear_op = openfermion.get_sparse_operator(
                    hamiltonian_qubit_op)

        # The variance bound is the squared one-norm of the coefficients,
        # omitting the constant term
        one_norm_minus_constant = (
                hamiltonian_qubit_op.induced_norm(order=1)
                - abs(hamiltonian_qubit_op.constant))
        self.variance_bound = one_norm_minus_constant**2

    def value(self,
              trial_result: Union[cirq.TrialResult,
                                  cirq.google.XmonSimulateTrialResult]
              ) -> float:
        # TODO implement support for TrialResult (compute energy given
        #      measurements)
        if not isinstance(trial_result, cirq.google.XmonSimulateTrialResult):
            raise NotImplementedError(
                    "Don't know how to compute the value of a TrialResult that "
                    "is not an XmonSimulateTrialResult.")
        return openfermion.expectation(
                self._hamiltonian_linear_op, trial_result.final_state).real

    def noise(self, cost: Optional[float]=None) -> float:
        """A sample from a normal distribution with mean 0.

        The variance of the distribution is equal to L^2 / cost, where L is the
        squared one-norm of the coefficients of the Pauli terms in the
        Jordan-Wigner transformed Hamiltonian. This gives an estimate of the
        variance of an energy measurement with a certain measurement strategy;
        see arXiv:1801.03524 for a derivation.
        """
        if cost is None:
            return 0.0
        return numpy.random.normal(
                loc=0.0, scale=numpy.sqrt(self.variance_bound / cost))

    def noise_bounds(self,
                     cost: float,
                     confidence: Optional[float]=None
                     ) -> Tuple[float, float]:
        """Exact or approximate bounds on noise in the objective function.

        Returns a tuple (a, b) such that the integral from a to b of the
        normal distribution that the noise is sampled from is equal to the
        confidence level. The variance of the normal distribution is inversely
        proportional to the cost provided, so providing a higher cost will
        yield a smaller interval, corresponding to a tighter bound on the noise.

        If confidence is not specified, a default value of .99 is used.
        """
        if confidence is None:
            confidence = 0.99

        if not 0 < confidence < 1:
            raise ValueError('The confidence in the noise bound must be '
                             'between 0 and 1.')

        sigmas = scipy.special.erfinv(confidence) * numpy.sqrt(2)
        magnitude_bound = sigmas * numpy.sqrt(self.variance_bound / cost)
        return -magnitude_bound, magnitude_bound
