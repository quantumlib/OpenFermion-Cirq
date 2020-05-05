.. currentmodule:: openfermioncirq

API Reference
=============

Gates
--------

Two-Qubit Gates
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    FermionicSwapGate
    XXYYGate
    YXXYGate
    ZZGate
    FSWAP
    XXYY
    YXXY
    ZZ

Three-Qubit Gates
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    ControlledXXYYGate
    ControlledYXXYGate
    Rot111Gate
    CXXYY
    CYXXY
    CCZ

Fermionic Simulation Gates
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    fermionic_simulation_gates_from_interaction_operator
    ParityPreservingFermionicGate
    QuadraticFermionicSimulationGate
    CubicFermionicSimulationGate
    QuarticFermionicSimulationGate


Primitives
----------

.. autosummary::
    :toctree: generated/

    bogoliubov_transform
    ffft
    prepare_gaussian_state
    prepare_slater_determinant
    swap_network


Hamiltonian Simulation
----------------------

.. autosummary::
    :toctree: generated/

    simulate_trotter
    trotter.TrotterStep
    trotter.TrotterAlgorithm
    trotter.LINEAR_SWAP_NETWORK
    trotter.SPLIT_OPERATOR
    trotter.LOW_RANK


Trotter Algorithms
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    trotter.LinearSwapNetworkTrotterAlgorithm
    trotter.SplitOperatorTrotterAlgorithm
    trotter.LowRankTrotterAlgorithm


Variational Algorithms
----------------------

.. autosummary::
    :toctree: generated/

    VariationalAnsatz
    VariationalStudy
    HamiltonianVariationalStudy

Variational Ansatzes
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    SwapNetworkTrotterAnsatz
    SplitOperatorTrotterAnsatz


Optimization
------------

.. autosummary::
    :toctree: generated/

    optimization.OptimizationAlgorithm
    optimization.OptimizationParams
    optimization.OptimizationResult
    optimization.OptimizationTrialResult
    optimization.ScipyOptimizationAlgorithm
    optimization.COBYLA
    optimization.L_BFGS_B
    optimization.NELDER_MEAD
    optimization.SLSQP
