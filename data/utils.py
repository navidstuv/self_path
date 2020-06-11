from albumentations.augmentations.transforms import Resize
import numpy as np
import cv2

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

    :param big_image:
    :return:
    """

    jig = np.zeros((128, 128, 3))
    img_40 = center_crop(big_image, 64, 64)

    img_20 = center_crop(big_image, 128, 128)
    img_20 = cv2.resize(img_20, (64, 64), interpolation=cv2.INTER_CUBIC)

    img_10 = center_crop(big_image, 256, 256)
    img_10 = cv2.resize(img_10, (64, 64), interpolation=cv2.INTER_CUBIC)

    img_5 = cv2.resize(big_image, (64, 64))

    order = np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12'], 1)
    if order=='1':
        jig[:64, :64, :] = img_40
        jig[64:, 64:, :] = img_20
        jig[:64, 64:, :] = img_10
        jig[64:, :64, :] = img_5
        jig_lbl = 0
    elif order=='2':
        jig[:64, :64, :] = img_40
        jig[64:, 64:, :] = img_10
        jig[:64, 64:, :] = img_20
        jig[64:, :64, :] = img_5
        jig_lbl = 1
    elif order=='3':
        jig[:64, :64, :] = img_40
        jig[64:, 64:, :] = img_5
        jig[:64, 64:, :] = img_20
        jig[64:, :64, :] = img_10
        jig_lbl = 2
    elif order=='4':
        jig[:64, :64, :] = img_5
        jig[64:, 64:, :] = img_40
        jig[:64, 64:, :] = img_20
        jig[64:, :64, :] = img_10
        jig_lbl = 3
    elif order=='5':
        jig[:64, :64, :] = img_10
        jig[64:, 64:, :] = img_5
        jig[:64, 64:, :] = img_20
        jig[64:, :64, :] = img_40
        jig_lbl = 4
    elif order=='6':
        jig[:64, :64, :] = img_20
        jig[64:, 64:, :] = img_5
        jig[:64, 64:, :] = img_10
        jig[64:, :64, :] = img_40
        jig_lbl = 5
    elif order=='7':
        jig[:64, :64, :] = img_20
        jig[64:, 64:, :] = img_10
        jig[:64, 64:, :] = img_40
        jig[64:, :64, :] = img_5
        jig_lbl = 6
    elif order=='8':
        jig[:64, :64, :] = img_10
        jig[64:, 64:, :] = img_40
        jig[:64, 64:, :] = img_5
        jig[64:, :64, :] = img_20
        jig_lbl = 7
    elif order=='9':
        jig[:64, :64, :] = img_10
        jig[64:, 64:, :] = img_10
        jig[:64, 64:, :] = img_10
        jig[64:, :64, :] = img_10
        jig_lbl = 8
    elif order=='10':
        jig[:64, :64, :] = img_20
        jig[64:, 64:, :] = img_20
        jig[:64, 64:, :] = img_20
        jig[64:, :64, :] = img_20
        jig_lbl = 9
    elif order=='11':
        jig[:64, :64, :] = img_40
        jig[64:, 64:, :] = img_40
        jig[:64, 64:, :] = img_40
        jig[64:, :64, :] = img_40
        jig_lbl = 10
    elif order=='12':
        jig[:64, :64, :] = img_5
        jig[64:, 64:, :] = img_5
        jig[:64, 64:, :] = img_5
        jig[64:, :64, :] = img_5
        jig_lbl = 11
    return jig, jig_lbl



