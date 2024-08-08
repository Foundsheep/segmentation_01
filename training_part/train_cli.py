import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI
from model_loader import SPRSegmentModel
from data_loader import SPRDataModule
from arg_parser import get_args
import sys
import datetime


# def main(args):
#     print(".................................")
#     print(f"The provided arguments are\n\t {sys.argv[1:]}")
#     print(".................................")

#     cli = LightningCLI(
#         model_class=SPRSegmentModel,
#         datamodule_class=SPRDataModule,
#         run=False,
#         seed_everything_defaut=123,

#     )

#     lightning_model = SPRSegmentModel(model_name=args.model_name,
#                                       loss_name=args.loss_name,
#                                       optimizer_name=cli.model.optimizer_name,
#                                       use_early_stop=cli.model.use_early_stop,
#                                       momentum=cli.model.momentum,
#                                       weight_decay=cli.model.weight_decay
#                                       )

#     cli.trainer.fit()

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print(".................................")
    print(f"The provided arguments are\n\t {args}")
    print(".................................")

    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    if args.checkpoint_dir:
        model = SPRSegmentModel.load_from_checkpoint(args.checkpoint_dir,
                                                     model_name=args.model_name,
                                                     loss_name=args.loss_name,
                                                     lr=args.lr,
                                                     optimizer_name=args.optimizer_name,
                                                     use_early_stop=args.use_early_stop,
                                                     momentum=args.momentum,
                                                     weight_decay=args.weight_decay)
        print(f"model loaded from [{args.checkpoint_dir}]")
    else:
        model = SPRSegmentModel(model_name=args.model_name,
                                loss_name=args.loss_name,
                                optimizer_name=args.optimizer_name,
                                lr=args.lr,
                                use_early_stop=args.use_early_stop,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    dm = SPRDataModule(root=args.root,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle,
                       dl_num_workers=args.dl_num_workers,
                       data_split=args.data_split,
                       train_ratio=args.train_ratio,
                       val_ratio=args.val_ratio,
                       test_ratio=args.test_ratio)

    trainer = L.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.train_log_folder + f"/{timestamp}_{args.model_name}"
    )


    trainer.fit(model=model,
                datamodule=dm)
    trainer.test(ckpt_path="best",
                 datamodule=dm)

# def main(args):
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

#     # GPU performance increases!
#     torch.set_float32_matmul_precision('medium')

#     print(f"=== root: [{args.root}]")
#     trainer = L.Trainer(accelerator="gpu" if Config.DEVICE == "cuda" else Config.DEVICE,
#                         min_epochs=args.min_epochs,
#                         max_epochs=args.max_epochs,
#                         log_every_n_steps=args.log_every_n_steps,
#                         inference_mode=True,
#                         default_root_dir=args.train_log_folder + f"/{timestamp}_{args.model_name}_{args.loss_fn}_epochs{args.max_epochs}_batch{args.batch_size}",
#                         callbacks=[early_stop_callback] if args.use_early_stop else [])
    
#     # dataloaders
#     train_dl = load_data(root=args.root + "/train",
#                          is_train=True,
#                          shuffle=args.shuffle,
#                          batch_size=args.batch_size,
#                          num_workers=args.dl_num_workers)
#     val_dl = load_data(root=args.root + "/val",
#                        is_train=False,
#                        shuffle=False,
#                        batch_size=args.batch_size,
#                        num_workers=args.dl_num_workers)
#     test_dl = load_data(root=args.root + "/test",
#                         is_train=False,
#                         shuffle=False,
#                         batch_size=args.batch_size,
#                         num_workers=args.dl_num_workers)


#     # from checkpoint dir
#     if args.checkpoint_dir:
#         model = SPRSegmentModel.load_from_checkpoint(args.checkpoint_dir,
#                                                      model_name=args.model_name,
#                                                      loss_fn=args.loss_fn,
#                                                      optimizer=args.optimizer)
#         print(f"model loaded from [{args.checkpoint_dir}]")
#     else:
#         model = SPRSegmentModel(args.model_name, args.loss_fn, args.optimizer)
#     # model.to(Config.DEVICE)

#     trainer.fit(model=model,
#                 train_dataloaders=train_dl,
#                 val_dataloaders=val_dl)
    
#     if test_dl is not None:
#         trainer.test(model, test_dl)


if __name__ == "__main__":
    args = get_args()
    main(args)