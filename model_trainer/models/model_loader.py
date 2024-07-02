from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import segmentation_models_pytorch as smp

def load_model(model_name):
    if model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
        )
    elif model_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=4,
        )
    elif model_name == "SegFormer":
        model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=4,
        )
    else:
        print(f"Model name is wrong. {model_name = }")
    return model


class SPRSegmentModel(L.LightningModule):
    def __init__(self, model_name):
        self.model = load_model(model_name)

    def training_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer