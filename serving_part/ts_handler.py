import torch
from pathlib import Path
import numpy as np
# import sys
# sys.path.append(str(Path(__file__).absolute().parent.parent))

from image_utils import erase_coloured_text_and_lines, get_transforms, get_label_info
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import datetime
import io

class SPRModelHandler(BaseHandler):
    def __init__(self):
        # super().__init__()
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        
    def initialize(self, context):
        self._context = context
        self.initialized = True
    
    def preprocess(self, data):
        transforms = get_transforms(is_train=False)
        print("===============")
        print(data)
        print(type(data))
        print("===============")
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")
            
        
        print("===============")
        print(image)
        print(type(image))
        print("===============")
        if isinstance(image, bytearray):
            image = np.array(Image.open(io.BytesIO(image)))
        
        transformed = transforms(image=np.asarray(image))
        image = transformed["image"]
        return image
    
    def inference(self, model_input):
        return self.model.forward(model_input)

    def postprocess(self, inference_output):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        out = F.softmax(inference_output, dim=1) # N, C, H, W
        out = out.argmax(1) # N, H, W
        out_2d = out.squeeze(0).cpu().numpy().astype(np.uint8) # H, W
        out_3d = np.stack([out_2d] * 3, axis=2) # H, W, C
        
        # colouring
        labeltxt_path = "./model_store/labelmap.txt"
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
        
        # return pixels
        return_dict = {}
        predicted_classes = np.unique(out_2d)
        total_pixels = out_2d.size
        for predicted_cl in predicted_classes:
            count = np.count_nonzero(out_2d == predicted_cl)
            return_dict[predicted_cl]
        return return_dict
    
    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)