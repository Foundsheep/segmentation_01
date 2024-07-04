import torch
import lightning as L
from models.model_loader import SPRSegmentModel
from data_loader import load_data
from arg_parser import get_args
from configs import Config
import datetime


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    print(f"=== root: [{args.root}]")
    trainer = L.Trainer(accelerator="gpu" if Config.DEVICE == "cuda" else Config.DEVICE,
                        min_epochs=args.min_epochs,
                        max_epochs=args.max_epochs,
                        log_every_n_steps=args.log_every_n_steps,
                        default_root_dir=args.train_log_folder + f"/{timestamp}")
    
    # dataloaders
    train_dl = load_data(root=args.root,
                         is_train=True,
                         shuffle=args.shuffle,
                         batch_size=args.batch_size,
                         num_workers=args.dl_num_workers)
    val_dl = load_data(root=args.root,
                       is_train=False,
                       shuffle=False,
                       batch_size=args.batch_size,
                       num_workers=args.dl_num_workers)

    model = SPRSegmentModel(args.model_name, args.loss_fn, args.optimizer)
    # model.to(Config.DEVICE)

    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl)


if __name__ == "__main__":
    args = get_args()
    main(args)