.. image:: https://github.com/quantumlib/openfermion-cirq/blob/master/dev_tools/logo.svg

**Alpha Release**. `OpenFermion <http://openfermion.org>`_ is an open source library for
obtaining and manipulating representations of fermionic systems (including
quantum chemistry) for simulation on quantum computers.
`Cirq <https://github.com/quantumlib/Cirq>`_ is an open source library for
writing, manipulating, and optimizing quantum circuits and running them
against quantum computers and simulators. OpenFermion-Cirq extends the functionality of
OpenFermion by providing routines and tools for using Cirq to compile and compose circuits
for quantum simulation algorithms.

.. image:: https://travis-ci.com/quantumlib/OpenFermion-Cirq.svg?token=7FwHBHqoxBzvgH51kThw&branch=master
  :target: https://travis-ci.com/quantumlib/OpenFermion-Cirq
  :alt: Build Status

.. image:: https://badge.fury.io/py/openfermioncirq.svg
    :target: https://badge.fury.io/py/openfermioncirq

.. image:: https://img.shields.io/badge/python-2.7%2C%203.5-brightgreen.svg

Getting started
===============

Installing OpenFermion-Cirq requires pip. Make sure that you are using an up-to-date version of it.
Once installation is complete, be sure to take a look at our
`ipython tutorials
<https://github.com/quantumlib/OpenFermion-Cirq/blob/master/examples>`__
and
`code documentation
<https://openfermion-cirq.readthedocs.io/en/latest/>`__.

Installation
------------

To install the latest PyPI releases as libraries (in user mode):

.. code-block:: bash

  python -m pip install --user openfermioncirq


Developer install
-----------------

To install the latest versions of OpenFermion, Cirq and OpenFermion-Cirq (in development mode):

.. code-block:: bash

  git clone https://github.com/quantumlib/OpenFermion-Cirq
  cd OpenFermion-Cirq
  python -m pip install -e .


How to contribute
=================

We'd love to accept your contributions and patches to OpenFermion-Cirq.
There are a few small guidelines to follow which you can read about
`here <https://github.com/quantumlib/OpenFermion-Cirq/blob/master/CONTRIBUTING.md>`_.

How to cite
===========
When using OpenFermion-Cirq for research projects, please cite:

    Jarrod R. McClean, Kevin J. Sung, Ian D. Kivlichan, Yudong Cao,
    Chengyu Dai, E. Schuyler Fried, Craig Gidney, Brendan Gimby,
    Pranav Gokhale, Thomas Häner, Tarini Hardikar, Vojtĕch Havlíček,
    Oscar Higgott, Cupjin Huang, Josh Izaac, Zhang Jiang, Xinle Liu,
    Sam McArdle, Matthew Neeley, Thomas O'Brien, Bryan O'Gorman, Isil Ozfidan,
    Maxwell D. Radin, Jhonathan Romero, Nicholas Rubin, Nicolas P. D. Sawaya,
    Kanav Setia, Sukin Sim, Damian S. Steiger, Mark Steudtner, Qiming Sun,
    Wei Sun, Daochen Wang, Fang Zhang and Ryan Babbush.
    *OpenFermion: The Electronic Structure Package for Quantum Computers*.
    `arXiv:1710.07629 <https://arxiv.org/abs/1710.07629>`__. 2017.

We are happy to include future contributors as authors on later releases.

Alpha Disclaimer
================

**OpenFermion-Cirq is currently in alpha.**
We are still making breaking changes.
We *will* break your code when we make new releases.
We recommend that you target a specific version of OpenFermion-Cirq, and periodically bump to the latest release.
That way you have control over when a breaking change affects you.

Copyright 2018 The OpenFermion Developers.
This is not an official Google product.
