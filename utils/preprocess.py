import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs import Config

def erase_coloured_text_and_lines(img_path):
    # 1. 하늘색 HSV : 186, 98%, 95%
    # 2. 녹색 HSV : 118, 98%, 95%
    #               118, 98%, 65%
    #               118, 98%, 35%
    # 3. 빨강 HSV : 0, 98%, 65%
    img = cv2.imread(img_path)
    image = img.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 위의 기준보다 좀 더 널럴하게 Saturation 맞췄음. 아닐 경우 잘 인지 안 됨
    s_max = 255
    v_max = 255
    lower = np.array([0, s_max*0.60, v_max*0.30])
    upper = np.array([200, s_max*0.98, v_max*0.98])
    # lower = np.array([0, 128, 128])
    # upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((9, 9), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    dst = cv2.inpaint(image, dilated_mask, 5, cv2.INPAINT_NS)

    return dst


def get_transforms(is_train):
    if is_train:
        transforms = A.Compose([
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
            A.HorizontalFlip(),
            A.Blur(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    return transforms 