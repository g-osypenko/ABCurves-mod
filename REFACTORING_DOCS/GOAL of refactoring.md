# Goal of Refactoring

The goal of this refactoring is pure trajectory generation and control: adapting ABCurves into a closed-loop Receding Horizon trajectory generator.

The task is focused strictly on trajectory properties, mathematical continuity ($C^1$ splicing between sequential trajectory segments), dynamic replanning towards a moving target, and biomechanical motion invariants (Two-Thirds Power Law, minimum-jerk profile, Sigma-Lognormal velocity curves, and 8–12 Hz physiological tremor). The library operates purely as an algorithmic trajectory engine without dependencies on detection or input components.
