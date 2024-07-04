import argparse
from configs import Config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME)
    parser.add_argument("--loss_fn", type=str, default=Config.LOSS_FN)
    parser.add_argument("--optimizer", type=str, default=Config.OPTIMIZER)
    parser.add_argument("--checkpoint_path", type=str, default=Config.CHECKPOINT_PATH)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--labelmap_txt_path", type=str, default=Config.LABELMAP_TXT_PATH)

    args = parser.parse_args()
    return args