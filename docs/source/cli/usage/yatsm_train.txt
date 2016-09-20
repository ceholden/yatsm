$ yatsm train --help
Usage: yatsm train [OPTIONS] <config> <classifier_config> <model>

  Train a classifier from ``scikit-learn`` on YATSM output and save result
  to file <model>. Dataset configuration is specified by <yatsm_config> and
  classifier and classifier parameters are specified by <classifier_config>.

Options:
  --kfold INTEGER  Number of folds in cross validation (default: 3)
  --seed INTEGER   Random number generator seed
  --plot           Show diagnostic plots
  --diagnostics    Run K-Fold diagnostics
  --overwrite      Overwrite output model file
  --help           Show this message and exit.
