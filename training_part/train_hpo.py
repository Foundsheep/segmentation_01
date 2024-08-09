import torch
import lightning as L
from model_loader import SPRSegmentModel
from data_loader import SPRDataModule
from arg_parser import get_args

from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from pathlib import Path
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

def train_func(args):    
    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    dm = SPRDataModule(root=args["root"],
                       batch_size=args["batch_size"],
                       shuffle=args["shuffle"],
                       dl_num_workers=args["dl_num_workers"],
                       data_split=args["data_split"],
                       train_ratio=args["train_ratio"],
                       val_ratio=args["val_ratio"],
                       test_ratio=args["test_ratio"])
    
    model = SPRSegmentModel(model_name=args["model_name"],
                            loss_name=args["loss_name"],
                            optimizer_name=args["optimizer_name"],
                            lr=args["lr"],
                            use_early_stop=args["use_early_stop"],
                            momentum=args["momentum"],
                            weight_decay=args["weight_decay"])

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def main_tune(args):
    reset_dir_for_ray_session = str(Path(__file__).parent.absolute())
    ray.init(_temp_dir=reset_dir_for_ray_session)

    search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([4, 6, 8]),
    }

    search_space.update(vars(args))    

    # scaling_config = ScalingConfig(
    #     num_workers=2, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    # )

    scaling_config = ScalingConfig(
        use_gpu=True
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )

    # Number of sampls from parameter space
    num_samples = 10
    
    scheduler = ASHAScheduler(max_t=args.max_epochs, grace_period=1, reduction_factor=2)

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

if __name__ == "__main__":
    args = get_args()
    results = main_tune(args)
    results.get_best_result(metric="val_loss", mode="min")
