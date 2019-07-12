import PIL
from PIL import Image
import cv2
import numpy as np

img = cv2.imread('/home/akanksha/Desktop/img/file (1).jpeg')
res = cv2.resize(img, dsize=(2,2), interpolation=cv2.INTER_CUBIC)

