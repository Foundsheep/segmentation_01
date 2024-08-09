import torch
from pathlib import Path

class Config():
    # preprocess
    RESIZED_HEIGHT = 640
    RESIZED_WIDTH = 960

    # model
    MODEL_NAME = "UnetPlusPlus"
    LOSS_FN = "DiceLoss"
    OPTIMIZER = "Adam"

    # train loop
    NUM_CLASSES = 5
    MAX_EPOCHS = 1
    MIN_EPOCHS = 1
    SHUFFLE = True
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DL_NUM_WORKERS = 2 # TODO: change later to adapt to the local circumstance
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = str(Path(__file__).absolute().parent / "training_part")

    # inference
    CHECKPOINT_PATH = "../serving_part/20240704_165444/lightning_logs/version_0/checkpoints/epoch=4-step=10.ckpt"
    LABELMAP_TXT_PATH = "../training_part/datasets/spr_sample_01/annotated/labelmap.txt"

    # code
    SUCCESS = 1
    FAILURE = 0