import os

import optuna
from lightning import LIghtningApp
from lightning_hpo import Sweep
from lightning_hpo.algorithm.optuna import OptunaAlgorithm
from lightning_hpo.distributions.distributions import (
    Categorical,
    IntUniform,
    LogUniform,
)

app = LIghtningApp(
    Sweep(
        script_path=os.path.join(os.path.dirname(__file__), "./train.py"),
        total_experiments=3,
        distributions={
            "max_epochs": IntUniform(2, 4)
        },
        algorithm=OptunaAlgorithm(optuna.craete_study(direction="maxtimize")),
        framework="pytorch_lightning",
    )
)