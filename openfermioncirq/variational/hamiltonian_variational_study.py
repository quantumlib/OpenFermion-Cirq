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

from typing import Any, Dict, Optional, Union

import numpy
import scipy.special

import cirq
import openfermion

from openfermioncirq.variational.ansatz import VariationalAnsatz

from openfermioncirq.variational.study import VariationalStudy


class HamiltonianVariationalStudy(VariationalStudy):
    """A study with a value and noise model associated with a Hamiltonian.

    The value of the ansatz is the expectation value of the Hamiltonian in the
    state that results from applying the circuit.

    The noise and cost models are as follows:
        Cost model: Cost corresponds to the number of measurement samples taken.
        Noise model: Noise is sampled from a normal distribution with mean 0 and
            variance inversely proportional to the number of measurement
            samples.

    Attributes:
        hamiltonian: The Hamiltonian associated with the ansatz, represented
            as a FermionOperator, QubitOperator, InteractionOperator, or
            DiagonalCoulombHamiltonian. If the input is not a QubitOperator,
            then the Jordan-Wigner transform is used.
    """

    def __init__(self,
                 name: str,
                 ansatz: VariationalAnsatz,
                 hamiltonian: Union[
                     openfermion.DiagonalCoulombHamiltonian,
                     openfermion.FermionOperator,
                     openfermion.InteractionOperator,
                     openfermion.QubitOperator],
                 preparation_circuit: Optional[cirq.Circuit]=None,
                 datadir: Optional[str]=None) -> None:
        self.hamiltonian = hamiltonian
        if isinstance(hamiltonian, openfermion.QubitOperator):
            hamiltonian_qubit_op = hamiltonian
        else:
            hamiltonian_qubit_op = openfermion.jordan_wigner(hamiltonian)
        self._variance_bound = hamiltonian_qubit_op.induced_norm(order=1)**2
        self._hamiltonian_linear_op = openfermion.LinearQubitOperator(
                hamiltonian_qubit_op)
        super().__init__(
                name,
                ansatz,
                preparation_circuit,
                datadir)

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
        sum of the absolute values of the coefficients of the Pauli terms in the
        Jordan-Wigner transformed Hamiltonian. This gives an estimate of the
        variance of an energy measurement with a certain measurement strategy;
        see arXiv:1801.03524 for a derivation.
        """
        if cost is None:
            return 0.0
        return numpy.random.normal(
                loc=0.0, scale=numpy.sqrt(self._variance_bound / cost))

    def noise_bound(self,
                    cost: Optional[float]=None,
                    confidence: float=0.99) -> float:
        """An approximate bound on the magnitude of the noise.

        This returns a value that gives a "likely" upper bound on the absolute
        value of noise produced with the specified cost. The probability that
        the bound is correct is specified by the `confidence` parameter, which
        must be strictly between 0 and 1. The default value is .99, which means
        that the returned value is a true upper bound with 99% probability.
        Lowering the confidence will give a smaller bound that is less likely to
        be correct.
        """
        if cost is None:
            return 0.0
        if not 0 < confidence < 1:
            raise ValueError('The confidence in the noise bound must be '
                             'between 0 and 1.')
        sigmas = scipy.special.erfinv(confidence) * numpy.sqrt(2)
        return sigmas * numpy.sqrt(self._variance_bound / cost)

    def _init_kwargs(self) -> Dict[str, Any]:
        """Arguments to pass to __init__ when re-loading the study."""
        return {'name': self.name,
                'ansatz': self._ansatz,
                'preparation_circuit': self._preparation_circuit,
                'hamiltonian': self.hamiltonian}
