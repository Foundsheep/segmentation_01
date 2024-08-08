import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.model_loader import SPRSegmentModel
from data_loader import load_data
from arg_parser import get_args
from configs import Config
from utils.data_seperator import seperate_data
import datetime

import optuna
from optuna_integration import PyTorchLightningPruningCallback

def main(args, trial: optuna.trial.Trial):

    # hyperparameter tuning
    batch_size = trial.suggest_int("batch_size", 4, 16)
    hparams = dict(batch_size=batch_size)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    # train test split
    if args.split_root:
        split_result = seperate_data(args.split_root,
                                     args.train_ratio,
                                     args.val_ratio,
                                     args.test_ratio)
        
        if split_result == Config.FAILURE:
            print("dataset seperation failed")
        elif split_result == Config.SUCCESS:
            print("dataset seperation succeeded")
 
        args.root = args.split_root

    # callbacks
    early_stop_callback = EarlyStopping(monitor="val_iou",
                                        patience=4,
                                        mode="max",
                                        verbose=True)
    hparams_callback = PyTorchLightningPruningCallback(trial, monitor="valid_loss")
    callbacks = []
    if args.use_early_stop:
        callbacks.append(early_stop_callback)
    if args.hparams_tuning:
        callbacks.append(hparams_callback)

    print(f"=== root: [{args.root}]")
    trainer = L.Trainer(accelerator="gpu" if Config.DEVICE == "cuda" else Config.DEVICE,
                        min_epochs=args.min_epochs,
                        max_epochs=args.max_epochs,
                        log_every_n_steps=args.log_every_n_steps,
                        inference_mode=True,
                        default_root_dir=args.train_log_folder + f"/{timestamp}_{args.model_name}_{args.loss_fn}_epochs{args.max_epochs}_batch{args.batch_size}",
                        callbacks=callbacks)
    
    # dataloaders
    train_dl = load_data(root=args.root + "/train",
                         is_train=True,
                         shuffle=args.shuffle,
                         batch_size=args.batch_size,
                         num_workers=args.dl_num_workers)
    val_dl = load_data(root=args.root + "/val",
                       is_train=False,
                       shuffle=False,
                       batch_size=args.batch_size,
                       num_workers=args.dl_num_workers)
    test_dl = load_data(root=args.root + "/test",
                        is_train=False,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.dl_num_workers)


    # from checkpoint dir
    if args.checkpoint_dir:
        model = SPRSegmentModel.load_from_checkpoint(args.checkpoint_dir,
                                                     model_name=args.model_name,
                                                     loss_fn=args.loss_fn,
                                                     optimizer=args.optimizer)
        print(f"model loaded from [{args.checkpoint_dir}]")
    else:
        model = SPRSegmentModel(args.model_name, args.loss_fn, args.optimizer)
    # model.to(Config.DEVICE)

    # hyperparameter log
    trainer.logger.log_hyperparams(hparams)

    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl)
    
    if test_dl is not None:
        trainer.test(model, test_dl)

    return trainer.callback_bmetrics["valid_loss"].item()

if __name__ == "__main__":
    args = get_args()
    main(args)