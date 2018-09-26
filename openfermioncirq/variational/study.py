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

"""The variational study class."""

from typing import (
        Any, Dict, Hashable, Iterable, List, Optional, Sequence, Type, Union,
        cast)

import collections
import itertools
import multiprocessing
import os
import pickle
import time

import numpy

import cirq

from openfermioncirq.variational import variational_black_box
from openfermioncirq.variational.ansatz import VariationalAnsatz
from openfermioncirq.variational.objective import VariationalObjective
from openfermioncirq.optimization import (
        OptimizationParams,
        OptimizationResult,
        OptimizationTrialResult,
        StatefulBlackBox)


class VariationalStudy:
    """The results from optimizing a variational ansatz.

    A VariationalStudy is used to facilitate optimizing the parameters
    of a variational ansatz. It contains methods for performing optimizations
    and saving and loading the results.

    Example::
        ansatz = SomeVariationalAnsatz()
        objective = SomeVariationalObjective()
        study = SomeVariationalStudy('my_study', ansatz, objective)
        optimization_params = OptimizationParams(
            algorithm=openfermioncirq.optimization.COBYLA,
            initial_guess=numpy.zeros(5))
        result = study.optimize(optimization_params, identifier='run0')
        print(result.optimal_value)         # prints a number
        print(result.params.initial_guess)  # prints the initial guess used
        study.save()  # saves the study with all results obtained so far

    Attributes:
        name: The name of the study.
        circuit: The circuit of the study, which is the preparation circuit, if
            any, followed by the ansatz circuit.
        ansatz: The ansatz being studied.
        objective: The objective function of interest.
        target: An optional target value one wants to achieve during
            optimization.
        trial_results: A dictionary of OptimizationTrialResults from
            optimization runs of the study. Key is the identifier used to
            label the run.
        num_params: The number of parameters in the circuit.
    """

    def __init__(self,
                 name: str,
                 ansatz: VariationalAnsatz,
                 objective: VariationalObjective,
                 preparation_circuit: Optional[cirq.Circuit]=None,
                 initial_state: Union[int, numpy.ndarray]=0,
                 target: Optional[float]=None,
                 black_box_type: Type[
                     variational_black_box.VariationalBlackBox]=
                     variational_black_box.UNITARY_SIMULATE,
                 datadir: Optional[str]=None) -> None:
        """
        Args:
            name: The name of the study.
            ansatz: The ansatz to study.
            objective: The objective function.
            preparation_circuit: A circuit to apply prior to the ansatz circuit.
                It should use the qubits belonging to the ansatz.
            initial_state: An initial state to use if the study circuit is
                run on a simulator.
            target: The target value one wants to achieve during optimization.
            black_box_type: The type of VariationalBlackBox to use for
                optimization.
            datadir: The directory to use when saving the study. The default
                behavior is to use the current working directory.
        """
        # TODO store results as a pandas DataFrame?
        self.name = name
        self.trial_results = collections.OrderedDict() \
                # type: Dict[Any, OptimizationTrialResult]
        self.target = target
        self.initial_state = initial_state
        self._ansatz = ansatz
        self._objective = objective
        self._preparation_circuit = preparation_circuit or cirq.Circuit()
        self._circuit = self._preparation_circuit + self._ansatz.circuit
        self._black_box_type = black_box_type
        self.datadir = datadir

    def optimize(self,
                 optimization_params: OptimizationParams,
                 identifier: Optional[Hashable]=None,
                 reevaluate_final_params: bool=False,
                 save_x_vals: bool=False,
                 repetitions: int=1,
                 seeds: Optional[Sequence[int]]=None,
                 use_multiprocessing: bool=False,
                 num_processes: Optional[int]=None
                 ) -> OptimizationTrialResult:
        """Perform an optimization run and save the results.

        Constructs a BlackBox that uses the study to perform function
        evaluations, then uses the given algorithm to optimize the BlackBox.
        The result is saved as an OptimizationTrialResult in the
        `trial_results` dictionary of the study under the key specified by
        `identifier`.

        The `cost_of_evaluate` argument affects how the BlackBox is constructed.
        If it is None, then the `evaluate` method of the BlackBox will call the
        `evaluate` method of the study. If it is not None, then the `evaluate`
        method of the BlackBox will call the `evaluate_with_cost` method of the
        study using this cost as input.

        Args:
            optimization_params: The parameters of the optimization run.
            identifier: An optional identifier for the run. This is used as
                the key to `self.results`, where results are saved. If not
                specified, it is set to a non-negative integer that is not
                already a key.
            reevaluate_final_params: Whether the optimal parameters returned
                by the optimization algorithm should be reevaluated using the
                `evaluate` method of the study and the optimal value adjusted
                accordingly. This is useful when the optimizer only has access
                to the noisy `evaluate_with_cost` method of the study (because
                `cost_of_evaluate` is set), but you are interested in the true
                noiseless value of the returned parameters.
            save_x_vals: Whether to save all points (x values) that the
                black box was queried at. Only used if the black box type is
                a subclass of StatefulBlackBox.
            repetitions: The number of times to run the optimization.
            seeds: Random number generator seeds to use for the repetitions.
                The default behavior is to randomly generate an independent seed
                for each repetition.
            use_multiprocessing: Whether to use multiprocessing to run
                repetitions in different processes.
            num_processes: The number of processes to use for multiprocessing.
                The default behavior is to use the output of
                `multiprocessing.cpu_count()`.

        Side effects:
            Saves the returned OptimizationTrialResult into the `trial_results`
            dictionary
        """
        return self.optimize_sweep([optimization_params],
                                   [identifier] if identifier else None,
                                   reevaluate_final_params,
                                   save_x_vals,
                                   repetitions,
                                   seeds,
                                   use_multiprocessing,
                                   num_processes)[0]

    def optimize_sweep(self,
                       param_sweep: Iterable[OptimizationParams],
                       identifiers: Optional[Iterable[Hashable]]=None,
                       reevaluate_final_params: bool=False,
                       save_x_vals: bool=False,
                       repetitions: int=1,
                       seeds: Optional[Sequence[int]]=None,
                       use_multiprocessing: bool=False,
                       num_processes: Optional[int]=None
                       ) -> List[OptimizationTrialResult]:
        """Perform multiple optimization runs and save the results.

        This is like `optimize`, but lets you specify multiple
        OptimizationParams to use for separate runs.

        Args:
            param_sweep: The parameters for the optimization runs.
            identifiers: Optional identifiers for the runs, one for each
                OptimizationParams object provided. This is used as the key
                to `self.results`, where results are saved. If not specified,
                then it will be set to a sequence of non-negative integers
                that are not already keys.
            reevaluate_final_params: Whether the optimal parameters returned
                by the optimization algorithm should be reevaluated using the
                `evaluate` method of the study and the optimal value adjusted
                accordingly. This is useful when the optimizer only has access
                to the noisy `evaluate_with_cost` method of the study (because
                `cost_of_evaluate` is set), but you are interested in the true
                noiseless value of the returned parameters.
            save_x_vals: Whether to save all points (x values) that the
                black box was queried at. Only used if the black box type is
                a subclass of StatefulBlackBox.
            repetitions: The number of times to run the algorithm for each
                set of optimization parameters.
            seeds: Random number generator seeds to use for the repetitions.
                The default behavior is to randomly generate an independent seed
                for each repetition.
            use_multiprocessing: Whether to use multiprocessing to run
                repetitions in different processes.
            num_processes: The number of processes to use for multiprocessing.
                The default behavior is to use the output of
                `multiprocessing.cpu_count()`.

        Side effects:
            Saves the returned OptimizationTrialResult into the results
            dictionary
        """
        if seeds is not None and len(seeds) < repetitions:
            raise ValueError(
                    "Provided fewer RNG seeds than the number of repetitions.")

        if identifiers is None:
            # Choose a sequence of integers as identifiers
            existing_integer_keys = {key for key in self.trial_results
                                     if isinstance(key, int)}
            if existing_integer_keys:
                start = max(existing_integer_keys) + 1
            else:
                start = 0
            identifiers = itertools.count(cast(int, start))  # type: ignore

        trial_results = []

        for identifier, optimization_params in zip(identifiers, param_sweep):

            result_list = self._get_result_list(
                    optimization_params,
                    reevaluate_final_params,
                    save_x_vals,
                    repetitions,
                    seeds,
                    use_multiprocessing,
                    num_processes)

            trial_result = OptimizationTrialResult(result_list,
                                                   optimization_params)
            trial_results.append(trial_result)

            # Save the result into the trial_results dictionary
            self.trial_results[identifier] = trial_result

        return trial_results


    def extend_result(self,
                      identifier: Hashable,
                      reevaluate_final_params: bool=False,
                      save_x_vals: bool=False,
                      repetitions: int=1,
                      seeds: Optional[Sequence[int]]=None,
                      use_multiprocessing: bool=False,
                      num_processes: Optional[int]=None
                      ) -> None:
        """Extend a result by repeating the run with the same parameters.

        The provided identifier is used as a key to the `trial_results`
        dictionary to retrieve an OptimizationTrialResult.
        The OptimizationParams associated with this trial result are used to
        perform additional repetitions of the optimization run. The results
        of these repetitions are appended to the stored OptimizationTrialResult.

        If there is no OptimizationTrialResult associated with the given
        identifier, an error is raised.

        Args:
            identifier: The identifier of the result to extend.
            reevaluate_final_params: Whether the optimal parameters returned
                by the optimization algorithm should be reevaluated using the
                `evaluate` method of the study and the optimal value adjusted
                accordingly. This is useful when the optimizer only has access
                to the noisy `evaluate_with_cost` method of the study (because
                `cost_of_evaluate` is set), but you are interested in the true
                noiseless value of the returned parameters.
            save_x_vals: Whether to save all points (x values) that the
                black box was queried at. Only used if the black box type is
                a subclass of StatefulBlackBox.
            repetitions: The number of repetitions to perform.
            seeds: Random number generator seeds to use for the repetitions.
                The default behavior is to randomly generate an independent seed
                for each repetition.
            use_multiprocessing: Whether to use multiprocessing to run
                repetitions in different processes.
            num_processes: The number of processes to use for multiprocessing.
                The default behavior is to use the output of
                `multiprocessing.cpu_count()`.

        Raises:
            KeyError: There was no existing result with the given identifier.
        """
        if identifier not in self.trial_results:
            raise KeyError('Could not find an existing result with the '
                           'identifier {}.'.format(identifier))

        optimization_params = self.trial_results[identifier].params

        result_list = self._get_result_list(
                optimization_params,
                reevaluate_final_params,
                save_x_vals,
                repetitions,
                seeds,
                use_multiprocessing,
                num_processes)

        self.trial_results[identifier].extend(result_list)

    def _get_result_list(
            self,
            optimization_params,
            reevaluate_final_params: bool,
            save_x_vals: bool,
            repetitions: int=1,
            seeds: Optional[Sequence[int]]=None,
            use_multiprocessing: bool=False,
            num_processes: Optional[int]=None
            ) -> List[OptimizationResult]:

        if use_multiprocessing:
            if num_processes is None:
                num_processes = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(num_processes)
            try:
                arg_tuples = (
                    (
                        self.ansatz,
                        self.objective,
                        self._preparation_circuit,
                        self.initial_state,
                        optimization_params,
                        reevaluate_final_params,
                        save_x_vals,
                        seeds[i] if seeds is not None
                        else numpy.random.randint(4294967296),
                        self.ansatz.default_initial_params(),
                        self._black_box_type
                    )
                    for i in range(repetitions)
                )
                result_list = pool.map(_run_optimization, arg_tuples)
            finally:
                pool.terminate()
        else:
            result_list = []
            for i in range(repetitions):
                result = _run_optimization(
                    (
                        self.ansatz,
                        self.objective,
                        self._preparation_circuit,
                        self.initial_state,
                        optimization_params,
                        reevaluate_final_params,
                        save_x_vals,
                        seeds[i] if seeds is not None
                        else numpy.random.randint(4294967296),
                        self.ansatz.default_initial_params(),
                        self._black_box_type
                    )
                )
                result_list.append(result)

        return result_list

    def __str__(self) -> str:
        header = []   # type: List[str]
        details = []  # type: List[str]
        optimal_value = numpy.inf
        optimal_identifier = None  # type: Optional[Hashable]

        for identifier, result in self.trial_results.items():

            result_opt = result.optimal_value
            if result_opt < optimal_value:
                optimal_value = result_opt
                optimal_identifier = identifier

            details.append(
                    '    Identifier: {}'.format(
                        identifier)
            )
            details.append(
                    '        Optimal value: {}'.format(
                        result_opt)
            )
            details.append(
                    '        Number of repetitions: {}'.format(
                        result.repetitions)
            )
            details.append(
                    '        Optimal value 1st, 2nd, 3rd quartiles:'
            )
            details.append(
                    '            {}'.format(
                        list(result.data_frame['optimal_value'].quantile(
                            [.25, .5, .75])))
            )
            details.append(
                    '        Num evaluations 1st, 2nd, 3rd quartiles:')
            details.append(
                    '            {}'.format(
                        list(result.data_frame['num_evaluations'].quantile(
                            [.25, .5, .75]))))
            details.append(
                    '        Cost spent 1st, 2nd, 3rd quartiles:'
            )
            details.append(
                    '            {}'.format(
                        list(result.data_frame['cost_spent'].quantile(
                            [.25, .5, .75])))
            )
            details.append(
                    '        Time spent 1st, 2nd, 3rd quartiles:'
            )
            details.append(
                    '            {}'.format(
                        list(result.data_frame['time'].quantile(
                            [.25, .5, .75])))
            )

        header.append(
                'This study contains {} trial results.'.format(
                    len(self.trial_results)))
        header.append(
                'The optimal value found among all trial results is {}.'.format(
                    optimal_value))
        header.append(
                'It was found by the run with identifier {}.'.format(
                    repr(optimal_identifier)))
        header.append('Result details:')

        return '\n'.join(header + details)

    @property
    def circuit(self) -> cirq.Circuit:
        """The preparation circuit followed by the ansatz circuit."""
        return self._circuit

    @property
    def ansatz(self) -> VariationalAnsatz:
        """The ansatz associated with the study."""
        return self._ansatz

    @property
    def objective(self) -> VariationalObjective:
        """The objective associated with the study."""
        return self._objective

    @property
    def num_params(self) -> int:
        """The number of parameters of the ansatz."""
        return len(list(self.ansatz.params()))

    def value_of(self,
                 params: numpy.ndarray) -> float:
        """Determine the value of some parameters."""
        return self._black_box_type(
                self.ansatz,
                self.objective,
                self._preparation_circuit,
                self.initial_state).evaluate_noiseless(params)

    def _init_kwargs(self) -> Dict[str, Any]:
        """Arguments to pass to __init__ when re-loading the study.

        Subclasses that override __init__ may need to override this method for
        saving and loading to work properly.
        """
        return {'name': self.name,
                'ansatz': self.ansatz,
                'objective': self.objective,
                'preparation_circuit': self._preparation_circuit}

    def save(self) -> None:
        """Save the study to disk."""
        filename = '{}.study'.format(self.name)
        if self.datadir is not None:
            filename = os.path.join(self.datadir, filename)
            if not os.path.isdir(self.datadir):
                os.mkdir(self.datadir)
        with open(filename, 'wb') as f:
            pickle.dump(
                    (type(self), self._init_kwargs(), self.trial_results), f)

    @staticmethod
    def load(name: str, datadir: Optional[str]=None) -> 'VariationalStudy':
        """Load a study from disk.

        Args:
            name: The name of the study.
            datadir: The directory where the study file is saved.
        """
        if name.endswith('.study'):
            filename = name
        else:
            filename = '{}.study'.format(name)
        if datadir is not None:
            filename = os.path.join(datadir, filename)
        with open(filename, 'rb') as f:
            cls, kwargs, trial_results = pickle.load(f)
        study = cls(datadir=datadir, **kwargs)
        for key, val in trial_results.items():
            study.trial_results[key] = val
        return study


def _run_optimization(args) -> OptimizationResult:
    """Perform an optimization run and return the result."""
    (
            ansatz,
            objective,
            preparation_circuit,
            initial_state,
            optimization_params,
            reevaluate_final_params,
            save_x_vals,
            seed,
            default_initial_params,
            black_box_type
    ) = args

    stateful = issubclass(black_box_type, StatefulBlackBox)

    if stateful:
        black_box = black_box_type(
                ansatz=ansatz,
                objective=objective,
                preparation_circuit=preparation_circuit,
                initial_state=initial_state,
                cost_of_evaluate=optimization_params.cost_of_evaluate,
                save_x_vals=save_x_vals)
    else:
        black_box = black_box_type(  # type: ignore
                ansatz=ansatz,
                objective=objective,
                preparation_circuit=preparation_circuit,
                initial_state=initial_state,
                cost_of_evaluate=optimization_params.cost_of_evaluate)

    initial_guess = optimization_params.initial_guess
    initial_guess_array = optimization_params.initial_guess_array
    if initial_guess is None:
        initial_guess = default_initial_params
    if initial_guess_array is None:
        initial_guess_array = numpy.array([default_initial_params])

    numpy.random.seed(seed)
    t0 = time.time()
    result = optimization_params.algorithm.optimize(black_box,
                                                    initial_guess,
                                                    initial_guess_array)
    t1 = time.time()

    result.seed = seed
    result.time = t1 - t0
    if stateful:
        result.num_evaluations = black_box.num_evaluations
        result.cost_spent = black_box.cost_spent
        result.function_values = black_box.function_values
        result.wait_times = black_box.wait_times
    if reevaluate_final_params:
        result.optimal_value = black_box.evaluate_noiseless(
                result.optimal_parameters)

    return result
