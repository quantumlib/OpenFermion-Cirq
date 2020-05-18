API Reference
=============


Gates
--------

Two-Qubit Gates
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.FSWAP
    openfermioncirq.XXYY
    openfermioncirq.YXXY
    openfermioncirq.rot11
    openfermioncirq.FSwapPowGate
    openfermioncirq.Rxxyy
    openfermioncirq.Ryxxy
    openfermioncirq.Rzz
    openfermioncirq.XXYYPowGate
    openfermioncirq.YXXYPowGate

Three-Qubit Gates
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.CXXYY
    openfermioncirq.CYXXY
    openfermioncirq.rot111
    openfermioncirq.CRxxyy
    openfermioncirq.CRyxxy
    openfermioncirq.CXXYYPowGate
    openfermioncirq.CYXXYPowGate

Four-Qubit Gates
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.DoubleExcitation
    openfermioncirq.DoubleExcitationGate

Fermionic Simulation Gates
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.fermionic_simulation_gates_from_interaction_operator
    openfermioncirq.CubicFermionicSimulationGate
    openfermioncirq.ParityPreservingFermionicGate
    openfermioncirq.QuadraticFermionicSimulationGate
    openfermioncirq.QuarticFermionicSimulationGate


Primitives
----------

.. autosummary::
    :toctree: generated/

    openfermioncirq.bogoliubov_transform
    openfermioncirq.ffft
    openfermioncirq.prepare_gaussian_state
    openfermioncirq.prepare_slater_determinant
    openfermioncirq.swap_network


Hamiltonian Simulation
----------------------

.. autosummary::
    :toctree: generated/

    openfermioncirq.simulate_trotter
    openfermioncirq.trotter.LINEAR_SWAP_NETWORK
    openfermioncirq.trotter.LOW_RANK
    openfermioncirq.trotter.SPLIT_OPERATOR
    openfermioncirq.trotter.TrotterAlgorithm
    openfermioncirq.trotter.TrotterStep

Trotter Algorithms
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.trotter.LinearSwapNetworkTrotterAlgorithm
    openfermioncirq.trotter.LowRankTrotterAlgorithm
    openfermioncirq.trotter.SplitOperatorTrotterAlgorithm


Variational Algorithms
----------------------

.. autosummary::
    :toctree: generated/

    openfermioncirq.HamiltonianObjective
    openfermioncirq.VariationalAnsatz
    openfermioncirq.VariationalObjective
    openfermioncirq.VariationalStudy

Variational Ansatzes
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    openfermioncirq.LowRankTrotterAnsatz
    openfermioncirq.SplitOperatorTrotterAnsatz
    openfermioncirq.SwapNetworkTrotterAnsatz
    openfermioncirq.SwapNetworkTrotterHubbardAnsatz


Optimization
------------

.. autosummary::
    :toctree: generated/

    openfermioncirq.optimization.COBYLA
    openfermioncirq.optimization.L_BFGS_B
    openfermioncirq.optimization.NELDER_MEAD
    openfermioncirq.optimization.SLSQP
    openfermioncirq.optimization.BlackBox
    openfermioncirq.optimization.OptimizationAlgorithm
    openfermioncirq.optimization.OptimizationParams
    openfermioncirq.optimization.OptimizationResult
    openfermioncirq.optimization.OptimizationTrialResult
    openfermioncirq.optimization.ScipyOptimizationAlgorithm
    openfermioncirq.optimization.StatefulBlackBox
