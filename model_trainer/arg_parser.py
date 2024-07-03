import argparse
from configs import Config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME)
    parser.add_argument("--loss_fn", type=str, default=Config.LOSS_FN)
    parser.add_argument("--optimizer", type=str, default=Config.OPTIMIZER)
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS)
    parser.add_argument("--min_epochs", type=int, default=Config.MIN_EPOCHS)
    parser.add_argument("--shuffle", type=bool, default=Config.SHUFFLE)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--dl_num_workers", type=int, default=Config.DL_NUM_WORKERS)
    parser.add_argument("--log_every_n_steps", type=int, default=Config.LOG_EVERY_N_STEPS)
    parser.add_argument("--train_log_folder", type=str, default=Config.TRAIN_LOG_FOLDER)

    args = parser.parse_args()
    return args