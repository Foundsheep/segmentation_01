from torch import optim
import lightning as L
import segmentation_models_pytorch as smp
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import datetime

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from configs import Config
from torch.nn.functional import softmax

# TODO:
# 1. accuracy or recall for each class to be logged 
# 2. get model files in local host

class SPRSegmentModel(L.LightningModule):
    def __init__(self, model_name, loss_name, optimizer_name, lr, use_early_stop=False, momentum=0., weight_decay=0.):
        super().__init__()
        self.model_name = model_name
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_early_stop = use_early_stop
        self.model = self._load_model(self.model_name)

        if loss_name == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer_name == "AdamW":
            if weight_decay == 0:
                self.weight_decay = 0.01
            self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.save_hyperparameters(ignore=["loss_fn", "optimizer", "use_early_stop"])

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
    
    def forward(self, x):
        x = self._preprocess(x)
        x = x.to(Config.DEVICE)
        self.model.to(Config.DEVICE)
        return self.model(x)

    def _preprocess(self, img):

        # to torch.Tensor
        if isinstance(img, Image.Image):
            x = np.array(img)
            x = torch.from_numpy(x)
        elif isinstance(img, np.ndarray):
            x = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            x = img
        else:
            raise TypeError(f"image should be either PIL.Image.Image or numpy.ndarray. Input type is {type(img)}")

        # C, H, W
        if x.size()[0] != 3:
            x = x.permute(2, 0, 1)

        # dtype conversion
        if x.dtype != torch.float:
            x = x.type(torch.float)

        # dimension for batch
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x
    
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

        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss) 
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return self.optimizer
    
    def configure_callbacks(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callbacks = []
        if self.use_early_stop:
            early_stop = EarlyStopping(monitor="val_iou",
                                    patience=4,
                                    mode="max",
                                    verbose=True)
            callbacks.append(early_stop)

        checkpoint = ModelCheckpoint(monitor="val_loss",
                                     filename="{epoch}-{step}-{val_loss:.3f}-{val_iou:.3f}")
        callbacks.append(checkpoint)
        return callbacks