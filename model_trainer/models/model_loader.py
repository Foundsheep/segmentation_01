from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import segmentation_models_pytorch as smp
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from configs import Config
from torch.nn.functional import softmax


class SPRSegmentModel(L.LightningModule):
    def __init__(self, model_name, loss_fn, optimizer):
        super().__init__()
        self.model = self._load_model(model_name)

        if loss_fn == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def _load_model(self, model_name):
        model = None
        if model_name == "UnetPlusPlus":
            model = smp.UnetPlusPlus(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=Config.NUM_CLASSES,     # model output channels (number of classes in your dataset)
            )
        elif model_name == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=Config.NUM_CLASSES,
            )
        elif model_name == "SegFormer":
            model = smp.Unet(
                encoder_name="mit_b1",
                encoder_weights="imagenet",
                in_channels=3,
                classes=Config.NUM_CLASSES,
            )
        else:
            print(f"Model name is wrong. {model_name = }")
        return model

    def shared_step(self, batch, stage):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y.long())

        preds_softmax = softmax(preds, dim=1)
        preds_argmax = preds_softmax.argmax(dim=1)
        y_argmax = y.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(preds_argmax.long(), 
                                               y_argmax.long(),
                                               mode="multiclass",
                                               num_classes=Config.NUM_CLASSES)
        
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")

        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss) 
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        return self.optimizer