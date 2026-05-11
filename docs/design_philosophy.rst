.. _design-philosophy:

Design Philosophy
=================

This page explains the thinking behind metaheuristic‑designer: the problems it
addresses, the design decisions it makes, and the vision it works toward.  If you
are wondering *why* the library is built the way it is, this is the place to
start.

Motivation: a field in need of infrastructure
---------------------------------------------

The field of metaheuristics faces a well‑documented set of challenges.  Swan et
al. (2015) identified the need for *standardized, explicit, machine‑readable
descriptions of metaheuristics* to advance scientific progress through greater
communicability, reproducibility, and interoperability [Swan2015]_.  In a
subsequent paper, Swan et al. (2022) argued that the field suffers from
fragmentation and that what is needed is **truly extensible algorithm templates
that support reuse without modification**, together with **white‑box problem
descriptions** that let domain knowledge be injected generically.  With that
infrastructure in place, researchers can explore the design space of
metaheuristics systematically rather than through ad‑hoc one‑offs.

A related and equally pressing problem is the proliferation of “novel”
metaheuristics that are merely old algorithms wrapped in new metaphors.
Weyland (2010) proved that Harmony Search—a method inspired by jazz music
improvisation—is a special case of Evolution Strategies, with mathematically
identical selection, mutation, and recombination operators [Weyland2010]_.  He
concluded that research on the method was “fundamentally misguided” and that
future effort “could better be devoted to more promising areas” [Weyland2010]_.
Similarly, Camacho‑Villalón et al. (2019) demonstrated that the Intelligent
Water Drops algorithm is simply a particular instantiation of Ant Colony
Optimization, and that the natural metaphor of “water drops flowing in rivers”
is “unnecessary, misleading and based on unconvincing assumptions” [Camacho2019]_.
Sörensen (2015) surveyed this landscape more broadly, arguing that the
increasing use of metaphors as inspiration and justification for new methods
“is threatening to lead the area of metaheuristics away from scientific
rigor” [Soerensen2015]_.

Metaheuristic‑designer is a direct answer to these calls.  Its architecture
embodies the **algorithm template** philosophy: the abstract interfaces are
extensible templates, the encoding/objective separation provides white‑box
problem descriptions, and the modular composition lets users explore algorithm
design space.  By making every component explicit and replaceable, the library
encourages the kind of principled, comparable, and reproducible research that
the field has been asking for.


The algorithm template
----------------------

The central insight of this library is that a very large class of metaheuristics
share a common structure.  Whether you are writing a genetic algorithm, an
evolution strategy, simulated annealing, or particle swarm optimisation, the
algorithm follows the same pattern:

1. **Initialise** a set of tentative solutions.
2. **Repeat** until a stopping condition is met:

   * **Select** a subset of the current solutions to generate new ones.
   * **Apply perturbation operators** to the selected solutions to produce
     new candidate solutions.
   * **Choose** which solutions will form the next iteration’s set of solutions.
3. **Return** the best solution found during the run.

Some algorithms skip certain steps.  Single‑solution methods like hill climbing
or simulated annealing trivially select the only solution.  Random search drops
the perturbation step and simply generates new solutions independently.  But the
template is general enough to capture all of them, and the library reflects this
generality directly in its interfaces.

This template is not a claim about how algorithms *should* be designed—it is an
observation about how they *already* are, once the metaphors are stripped away.
Weyland put it clearly: “I would propose to treat heuristics from a purely
mathematical and technical way.  Metaphors and analogies might (or might not)
help in the construction of heuristics, but the resulting methods should be
treated from a mathematical and technical point of view” [Weyland2010]_.
Swan et al. (2015) made the same point earlier, noting that “metaphors often
inspire new metaheuristics, but without mathematical rigor, it can be hard to
tell if a new metaheuristic is really distinct from a familiar one” [Swan2015]_.

From template to interfaces
----------------------------

Each step of the algorithm template is represented by an abstract class in the
library.  This design has two purposes: it documents the contract that every
implementation must fulfil, and it allows any implementation to be swapped out
without changing anything else.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Template step
     - Library interface
   * - Initialisation
     - :class:`~metaheuristic_designer.initializer.Initializer` — generates
       the first set of tentative solutions.  Built‑in implementations include
       uniform, Gaussian, and permutation initializers.
   * - Representation
     - :class:`~metaheuristic_designer.encoding.Encoding` — transforms
       between the internal numerical representation used by operators and the
       representation understood by the objective function.  This separation
       is one of the library’s core design decisions: it means operators can
       be written once and reused across any problem, while the encoding
       handles the mapping.
   * - Selection (to generate new solutions)
     - :class:`~metaheuristic_designer.parent_selection.ParentSelection` —
       chooses which solutions will be used to produce new candidates without modification.
   * - Perturbation
     - :class:`~metaheuristic_designer.operator.Operator` — generates new
       candidate solutions by modifying existing ones.  Operators can be
       mutation, crossover, or more complex composite operations.
   * - Selection (to retain solutions)
     - :class:`~metaheuristic_designer.survivor_selection.SurvivorSelection` —
       decides which solutions are carried forward to the next iteration without modification.
   * - Iteration
     - :class:`~metaheuristic_designer.search_strategy.SearchStrategy` —
       combines the above components into a single iteration step.  Pre‑built
       strategies (GA, DE, PSO, ES, SA) are provided, and you can also
       assemble your own.
   * - Full loop
     - :class:`~metaheuristic_designer.algorithm.Algorithm` — runs the
       iteration until a stopping condition is met, managing reporting,
       history tracking, and checkpointing.

The interfaces are designed to be minimal.  For example, an operator only needs
to implement one method: accept a population and return a modified population.
This minimalism is deliberate—it means you can write new components as plain
Python functions and wrap them in the provided ``*FromLambda`` classes, without
inheriting from anything.

Composability and the “in the large” vision
--------------------------------------------

Swan et al. (2015) argued that making inputs and outputs explicit and avoiding
side‑effects “greatly facilitates automated assembly of metaheuristics,”
allowing software to “search for effective ways to combine metaheuristics
‘bottom‑up’, avoiding the human bias inherent in choosing specific
metaheuristics a priori” [Swan2015]_.  The same authors later identified
three conceptual pillars for moving the field forward: extensible algorithm
templates that support reuse without modification, white‑box problem
descriptions that let domain knowledge be injected generically, and
infrastructure that enables exploring the design space of metaheuristics
systematically.

Metaheuristic‑designer implements all three pillars:

* **Extensible templates**: Every abstract interface can be implemented by the
  user.  The library provides many built‑in implementations, but none of them
  are privileged—your own operator or selection method slots into the same
  place and participates in the same infrastructure.
* **White‑box problem descriptions**: The encoding layer decouples the
  optimization algorithm from the problem representation.  Operators work on a
  common numerical genotype; the encoding translates to whatever the objective
  function requires.  This makes the problem’s structure visible to the
  algorithm infrastructure, enabling generic tooling for visualization,
  parameter tuning, and automated algorithm configuration.
* **Design‑space exploration**: Because components are interchangeable, you
  can systematically vary one component (say, the mutation operator) while
  keeping everything else fixed, and compare results.  This turns algorithm
  design into a controlled experiment rather than a one‑off craft exercise.

A note on language
------------------

The library’s own class names—:class:`ParentSelection`,
:class:`SurvivorSelection`—are inherited from the evolutionary computation
literature and remain in place for recognisability.  However, in the
documentation we prefer optimisation‑native language: “tentative solutions”,
“candidate solutions”, “perturbation operators”, and “selecting which
solutions to retain”.  This follows the recommendation of Swan et al. (2015)
that we adopt “a purely functional description of metaheuristics — separate
from any metaphors that inspire them and with no hidden mechanisms” [Swan2015]_,
and echoes Weyland’s call to treat heuristics mathematically rather than
metaphorically.  Metaphors may inspire algorithm design, but they should not
obscure what the algorithm actually does.


References
----------

.. [Swan2022] Swan, J., Adriaensen, S., Brownlee, A. E. I., et al. (2022).
   `Metaheuristics “In the Large” <https://doi.org/10.1016/j.ejor.2021.05.042>`_.
   *European Journal of Operational Research*, 297(2), 393–406.

.. [Swan2015] Swan, J., Adriaensen, S., Bishr, M., et al. (2015).
   `A Research Agenda for Metaheuristic Standardization <http://www.cs.nott.ac.uk/~pszeo/docs/publications/research-agenda-metaheuristic.pdf>`_.
   *Proceedings of the XI Metaheuristics International Conference (MIC 2015)*.

.. [Soerensen2015] Sörensen, K. (2015).
   `Metaheuristics—the metaphor exposed <https://doi.org/10.1111/itor.12001>`_.
   *International Transactions in Operational Research*, 22(1), 3–18.

.. [Weyland2010] Weyland, D. (2010).
   `A Rigorous Analysis of the Harmony Search Algorithm: How the Research
   Community can be Misled by a “Novel” Methodology <https://people.idsia.ch/~weyland/harmony_search.pdf>`_.
   *International Journal of Applied Metaheuristic Computing*, 1(2), 50–60.

.. [Camacho2019] Camacho‑Villalón, C. L., Dorigo, M., & Stützle, T. (2019).
   `The intelligent water drops algorithm: why it cannot be considered a novel
   algorithm <https://doi.org/10.1007/s11721-019-00165-y>`_.
   *Swarm Intelligence*, 13, 173–192.

.. [Weyland2015] Weyland, D. (2015).
   `The Harmony Search Algorithm – My personal experience with this “novel”
   metaheuristic <http://www.dennisweyland.net/blog/?p=12>`_.
   Personal blog.