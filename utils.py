from cv2 import transform
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from PIL import Image
from torchvision import transforms as T

transform = T.ToTensor()


def load_image(path):
    return img_as_float32(io.imread(path))


def load_image_PIL(path):
    return transform(Image.open(path).convert('RGB')).cuda()


def save_img(save_path, img):
    Image.fromarray(np.array(img * 255).astype('uint8')).save(save_path, quality=95)