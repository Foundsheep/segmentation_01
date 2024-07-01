import cv2
import numpy as np
import matplotlib.pyplot as plt

def erase_coloured_text_and_lines(img_path):
    img_path = r"C:\Users\msi\Desktop\workspace\001_HM_SPR\04_datasets\1st\SPR2_BG0G46E(DEHG13598)_SABC1470(1.1t)_A365.0(3.0t)_002.jpg"

    img = cv2.imread(img_path)
    image = img.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 128, 128])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    dst = cv2.inpaint(image, dilated_mask, 5, cv2.INPAINT_NS)

    return dst