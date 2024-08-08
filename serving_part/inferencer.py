import sys
from pathlib import Path

print(Path.cwd().parent)
sys.path.append(str(Path.cwd().parent))

from training_part.model_loader import SPRSegmentModel
from utils.preprocess import get_transforms
import datetime
from PIL import Image
import numpy as np
from arg_parser import get_args
import torch.nn.functional as F


def get_label_info(labelmap_txt_path):
    def _make_mapping_dict(label_txt):
        label_to_name = {}
        name_to_label = {}
        label_to_rgb = {}

        for txt_idx, txt in enumerate(label_txt):
            divider_1 = txt.find(":")
            divider_2 = txt.find("::")

            label_name = txt[:divider_1]
            label_value = txt[divider_1+1:divider_2]
            rgb_values = list(map(int, label_value.split(",")))

            label_to_name[txt_idx] = label_name
            name_to_label[label_name] = txt_idx
            label_to_rgb[txt_idx] = rgb_values

        return label_to_name, name_to_label, label_to_rgb

    with open(labelmap_txt_path, "r") as f:
        label_txt = f.readlines()[1:]

    return _make_mapping_dict(label_txt)



def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model = SPRSegmentModel.load_from_checkpoint(args.checkpoint_path,
                                                 model_name=args.model_name,
                                                 loss_fn=args.loss_fn,
                                                 optimizer=args.optimizer)

    # TODO: inpaint preprocessing needed
    image = Image.open(args.image_path)

    transforms = get_transforms(is_train=False)
    transformed = transforms(image=np.asarray(image))
    image = transformed["image"]

    model.eval()
    out = model(image)
    
    # map the predicted classes to label
    label_to_name, name_to_label, label_to_rgb = get_label_info(args.labelmap_txt_path)

    out_softmaxed = F.softmax(out, dim=1) # N, C, H, W
    out_argmaxed = out_softmaxed.argmax(1) # N, H, W
    out_2dim = out_argmaxed.squeeze(0).cpu().numpy().astype(np.uint8) # H, W
    out_3dim = np.stack([out_2dim] * 3, axis=2)

    # colouring
    for class_idx, rgb in label_to_rgb.items():
        x, y = np.where(out_2dim == class_idx)
        out_3dim[x, y, :] = rgb

    # save
    out_img = Image.fromarray(out_3dim)

    file_name = Path(args.image_path).parts[-1]
    save_folder = Path(f"./inference_result_{timestamp}")
    if not save_folder.exists():
        save_folder.mkdir()
        print(f"[{str(save_folder)} is made]")
    
    save_path = save_folder / Path(file_name)
    out_img.save(save_path)

if __name__ == "__main__":
    args = get_args()
    main(args)