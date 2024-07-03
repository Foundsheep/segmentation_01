import torch
from pathlib import Path

class Config():
    MODEL_NAME = "UnetPlusPlus"
    LOSS_FN = "DiceLoss"
    NUM_CLASSES = 4
    OPTIMIZER = "Adam"
    MAX_EPOCHS = 3
    MIN_EPOCHS = 2
    SHUFFLE = True
    BATCH_SIZE = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DL_NUM_WORKERS = 4 # TODO: change later to adapt to the local circumstance
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = "../model_server" if Path.cwd().parts[-1] == "model_trainer" else "./"