import os

import optuna
import lightning as L

from arg_parser import get_args
import train
from functools import partial

def wrapping_main() -> float:
    args = get_args()
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    objective = partial(train.main, args=args)
    study.optimize(objective, n_trials=10)

    print(f"[{len(study.trials)}] trials finished...")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"    Value: [{best_trial.value}]")
    print(f"    Params:")
    for k, v in best_trial.params.item():
        print(f"        k: {k}, v: {v}")


if __name__ == "__main__":
    wrapping_main()