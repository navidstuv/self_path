from albumentations.augmentations.transforms import Resize
import numpy as np
import cv2
import random
import itertools

def center_crop(img, crop_height, crop_width):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[y1:y2, x1:x2]
    return img

def get_center_crop_coords(height, width, crop_height, crop_width):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def jigsaw_res(big_image):
    """

    :param big_image: 128*128 image
    :return: jigmag image
    """
    # Just to check if the main image is 1024 or not,
    # for magnification task images of size 512x512 are extracted from WSIs
    if big_image.shape[0]==1024:
        big_image = center_crop(big_image, 512, 512)
    jig = np.zeros((128, 128, 3))
    img_40 = center_crop(big_image, 64, 64)
    img_20 = center_crop(big_image, 128, 128)
    img_20 = cv2.resize(img_20, (64, 64), interpolation=cv2.INTER_CUBIC)
    img_10 = center_crop(big_image, 256, 256)
    img_10 = cv2.resize(img_10, (64, 64), interpolation=cv2.INTER_CUBIC)
    img_5 = cv2.resize(big_image, (64, 64))
    list_imgs = [img_5, img_10, img_20, img_40]
    all_possible_permutations = list(itertools.permutations(list_imgs))
    order = random.sample(list(range(23)), 1)[0]
    jig[:64, :64, :] = all_possible_permutations[order][0]
    jig[64:, 64:, :] = all_possible_permutations[order][1]
    jig[:64, 64:, :] = all_possible_permutations[order][2]
    jig[64:, :64, :] = all_possible_permutations[order][3]
    jig_lbl = order
    return jig, jig_lbl



