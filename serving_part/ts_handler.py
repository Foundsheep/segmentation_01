import torch
from pathlib import Path
import numpy as np

import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import datetime
import io

# the below two files follow the path for /tmp/model/.../
# where all model_store files are copied to
from model_loader import SPRSegmentModel
from image_utils import erase_coloured_text_and_lines, get_transforms, get_label_info

class SPRModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        
    def initialize(self, context):
        self._context = context
        self.initialized = True
        
        # model init
        model_config = context.model_yaml_config
        model_name = model_config["model_name"]
        loss_name = model_config["loss_name"]
        lr = model_config["lr"]
        optimizer_name = model_config["optimizer_name"]
        self.model = SPRSegmentModel(
            model_name=model_name,
            loss_name=loss_name,
            optimizer_name=optimizer_name,
            lr=lr
        )
        self.model.eval()
        
        # properties
        properties = context.system_properties
        self.model_dir = properties["model_dir"]
        
    
    def preprocess(self, data):
        transforms = get_transforms(is_train=False)
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")
            
        if isinstance(image, bytearray):
            image = np.array(Image.open(io.BytesIO(image)))
        
        transformed = transforms(image=np.asarray(image))
        image = transformed["image"]
        return image
    
    def inference(self, model_input):
        return self.model.forward(model_input)

    # TODO: change the image size to its original
    def postprocess(self, inference_output):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        out = F.softmax(inference_output, dim=1) # N, C, H, W
        out = out.argmax(1) # N, H, W
        out_2d = out.squeeze(0).cpu().numpy().astype(np.uint8) # H, W
        out_3d = np.stack([out_2d] * 3, axis=2) # H, W, C
        
        # colouring
        labeltxt_path = self.model_dir + "/labelmap.txt"
        label_info = get_label_info(labeltxt_path=labeltxt_path)
        
        for class_num, rgb in label_info.items():
            x, y = np.where(out_2d == class_num)
            out_3d[x, y, :] = rgb
            
        # save
        out_img = Image.fromarray(out_3d)
        
        save_folder = Path(f"./inference_result_{timestamp}")
        if not save_folder.exists():
            save_folder.mkdir()
            print(f"[{str(save_folder)} is made]")

        save_path = save_folder / "output.png"
        out_img.save(save_path)
                
        # to return        
        output = np.expand_dims(out_3d, axis=0)
        output = output.tolist()
        return output
    
    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
    