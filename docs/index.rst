.. metaheuristic-designer documentation master file, created by
   sphinx-quickstart on Wed Jul 19 18:39:56 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to metaheuristic‑designer
==================================

Metaheuristic‑designer is a modular, object‑oriented framework for building,
testing, and analysing population‑based optimisation algorithms.  It is
designed to address the challenges identified in `Metaheuristics “In the
Large” <https://doi.org/10.1016/j.ejor.2021.05.042>`_ (Swan et al., 2022): the
field’s fragmentation, lack of reproducibility, and the need for common
protocols that let researchers explore the design space of metaheuristics
systematically.

Whether you want a standard genetic algorithm in one line or a custom hybrid
strategy assembled from scratch, this library provides the building blocks.
Its architecture follows the **algorithm template** philosophy advocated by
that paper and draws on the design principles of `Introduction to
Evolutionary Computing <https://doi.org/10.1007/978-3-662-44874-8>`_ by Eiben
and Smith.

.. For the architectural vision and the research context behind the library,
.. read the :doc:`Design philosophy <design_philosophy>` page.

.. _first-steps:

Where to Start
--------------

The documentation is organised by topics, pick the entry point that
matches what you want to do right now:

* **Quick start** – Run a ready‑to‑use algorithm in under a minute on the
  :doc:`Quick Start <quick_start>` page.
* **Simple subpackage** – Browse the full catalogue of :doc:`pre‑packaged algorithms
  <simple>` and see which encodings each supports.
* **Algorithm configuration** – Learn how to set stopping conditions, reporters,
  history trackers, and checkpointers in the
  :doc:`Algorithm Configuration <api_reference.algorithm_config>` guide.
* **All built‑in methods** – Detailed tables of :doc:`operators and selection
  methods <api_reference.methods>`, including all probability distributions.
* **Custom components** – Write your own operators, selection methods,
  encodings, and more with the
  :doc:`Extending the Framework <api_reference.custom_components>` guide.
* **Plot results** – Turn recorded history into convergence, diversity, and
  parameter evolution plots with the :doc:`Plotting Tutorial <api_reference.plotting>`.

.. * **Design philosophy** – Read about the architectural vision behind the
  library in the :doc:`Design philosophy <design_philosophy>` page.

.. _description:

Description
-----------

The library is built with a focus on **clean architecture** and
**composability**, while remaining practical for research and application.
When comparing algorithms, we recommend using the **number of objective
evaluations** rather than wall‑clock time to obtain a fair,
implementation‑independent measure.

.. _structure:

Structure
---------

Conceptual overview
~~~~~~~~~~~~~~~~~~~

Many optimisation algorithms follow the same general pattern:

1. **Initialise** a set of tentative solutions.
2. **Repeat** until a stopping condition is met:

   #. **Select** a subset of the current solutions to generate new ones.
   #. **Apply perturbation operators** to the selected solutions to produce
      new candidate solutions.
   #. **Choose** which solutions will form the next iteration’s population.
3. **Return** the best solution found during the run.

Key components
~~~~~~~~~~~~~~

The library provides an abstract interface for each step of this loop.  Every
component can be replaced independently, which makes it easy to experiment with
different algorithm variants.

* :class:`~metaheuristic_designer.ObjectiveFunc` — the function to optimise.
* :class:`~metaheuristic_designer.Initializer` — creates the initial set of
  tentative solutions.
* :class:`~metaheuristic_designer.Encoding` — transforms between the internal
  numerical representation (genotype) and the representation understood by the
  objective function.
* :class:`~metaheuristic_designer.Operator` — generates new candidate
  solutions by modifying existing ones (for example, adding random noise,
  recombining two solutions, or performing a local search step).
* :class:`~metaheuristic_designer.ParentSelection` /
  :class:`~metaheuristic_designer.SurvivorSelection` — choose which solutions
  are used to generate new ones and which ones are carried over to the next
  iteration.
* :class:`~metaheuristic_designer.SearchStrategy` — combines the above
  components into a single iteration step.
* :class:`~metaheuristic_designer.Algorithm` — runs the loop, tracks progress,
  manages stopping conditions, reporting, history, and checkpointing.

All of these have ready‑to‑use implementations in their respective
sub‑packages.  You can also supply your own components as plain Python
functions via the ``*FromLambda`` classes described in the
:doc:`Custom Components <api_reference.custom_components>` guide.

Indices and tables
==================

:ref:`genindex`

:ref:`search`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

    Quick Start <quick_start>
    Simple subpackage <simple>
    API reference <api_reference>
    Algorithm Configuration <api_reference.algorithm_config>
    Operators and selection methods <api_reference.methods>
    Custom components <api_reference.custom_components>
    Plotting Tutorial <api_reference.plotting>
    Module Details <auto/modules>

.. _references:

Further reading
---------------

* Swan, J., Adriaensen, S., Brownlee, A. E. I., et al. (2022).
  `Metaheuristics “In the Large” <https://doi.org/10.1016/j.ejor.2021.05.042>`_.
  *European Journal of Operational Research*, 297(2), 393–406.
* Eiben, A. E., & Smith, J. E. (2015).
  `Introduction to Evolutionary Computing <https://doi.org/10.1007/978-3-662-44874-8>`_.
  Springer.
