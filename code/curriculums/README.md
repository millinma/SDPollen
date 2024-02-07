## Curriculum Scores / Sample Difficulty

All scoring functions must implement the `run` method which performs the actual scoring of a given run configuration.
The `run` method can be called in parallel if the score consists of multiple runs (if `_use_slurm == True`).
Additionally, scoring functions can override the `preprocess` and `postprocess` methods to perform any preprocessing or postprocessing steps deviating from the default behavior defined in `AbstractScore`.

### Results of a Scoring Function

If a scoring function consists of multiple runs, the scores of the individual runs are averaged per example to obtain the final score.
The examples are ranked by sorting the scores in ascending order (can also be descending).
Finally, the scores are normalized to the range [0, 1] where 0 represents the easiest example and 1 the hardest.

### Bootstrapping ([Link](https://arxiv.org/pdf/1904.03626.pdf), [Link](https://arxiv.org/pdf/2012.03107.pdf))

Train a model on the full training dataset and use the loss of each example as the score.
