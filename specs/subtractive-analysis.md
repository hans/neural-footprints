Add a parallel negative analysis to this presentation. we want to show that a "subtractive analysis," commonly used to localize particular cognitive functions within systems neuro, fails to localize regions specific to high-level physical representations.

This analysis requires some more assumptions about the mapping between cognitive states and the neural data (i.e. for this to be fair we have to assume that some "regions" have relatively more resources allocated to function A vs. B).

## Demonstration of a subtractive analysis in this domain

We are looking for the neural representation of motion.
Method: first calibrate by identifying a low-level representation of objects. Compare neural activation in response to an empty scene vs. in response to a two-object scene. Set some activation threshold such that we detect primary “object regions.” Visualize the result as a “brain map” (here a 2D grid with weights resulting from the subtractive analysis).
Now use the same subtractive method on paired scenes. Each item of a scene pair has the same objects; one has motion and one does not. Use the calibrated threshold from before to detect a “motion region” which responds selectively to object motion.
Compare to some ground truth, based on e.g. the forward weights mapping from model state to brain state, and calculate true positives / false positives.