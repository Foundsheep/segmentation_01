import torch
from pathlib import Path

class Config():
    MODEL_NAME = "UnetPlusPlus"
    LOSS_FN = "DiceLoss"
    OPTIMIZER = "Adam"

    # train
    NUM_CLASSES = 4
    MAX_EPOCHS = 3
    MIN_EPOCHS = 2
    SHUFFLE = True
    BATCH_SIZE = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DL_NUM_WORKERS = 2 # TODO: change later to adapt to the local circumstance
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = "../model_server" if Path.cwd().parts[-1] == "model_trainer" else "./"
    RESIZED_HEIGHT = 640
    RESIZED_WIDTH = 960

    # inference
    CHECKPOINT_PATH = "../model_server/20240704_165444/lightning_logs/version_0/checkpoints/epoch=4-step=10.ckpt"
    LABELMAP_TXT_PATH = "../model_trainer/datasets/spr_sample_01/annotated/labelmap.txt"

    # code
    SUCCESS = 1
    FAILURE = 0