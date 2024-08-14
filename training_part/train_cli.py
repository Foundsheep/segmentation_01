import torch
import lightning as L
from arg_parser import get_args
import sys
import datetime

from model_loader import SPRSegmentModel
from data_loader import SPRDataModule
from configs import Config

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

    if args.backbone_name:
        default_root_dir = f"{args.train_log_folder}/{timestamp}_{args.model_name}_{args.backbone_name}_{args.loss_name}_batch{args.batch_size}_epoch{args.max_epochs}_lr{args.lr}"
    else:
        default_root_dir = f"{args.train_log_folder}/{timestamp}_{args.model_name}_{args.loss_name}_batch{args.batch_size}_epoch{args.max_epochs}_lr{args.lr}"

    trainer = L.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=default_root_dir            
    )


    trainer.fit(model=model,
                datamodule=dm)
    trainer.test(ckpt_path="best",
                 datamodule=dm)

    example_input = torch.randn(3, Config.TARGET_HEIGHT, Config.TARGET_WIDTH)
    script_model = model.to_torchscript(method="trace", example_inputs=example_input)
    
    torch.jit.save(script_model, f"{default_root_dir}/script_model.pt")
    print("Torch script model has been saved!")
    
if __name__ == "__main__":
    args = get_args()
    main(args)