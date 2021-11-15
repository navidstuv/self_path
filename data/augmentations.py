import random
import albumentations as A
import cv2
from albumentations.augmentations.functional import brightness_contrast_adjust
from albumentations.pytorch import ToTensor
from albumentations.augmentations.transforms import CenterCrop
class IndependentRandomBrightnessContrast(A.ImageOnlyTransform):
    """ Change brightness & contrast independently per channels """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super(IndependentRandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)

    def apply(self, img, **params):
        img = img.copy()
        for ch in range(img.shape[2]):
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            img[..., ch] = brightness_contrast_adjust(img[..., ch], alpha, beta)

        return img

def get_light_augmentations(image_size):
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0,p=0.3),
        A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.85), image_size[0]),
                          height=image_size[0],
                          width=image_size[1], p=0.3),
        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.2),
            IndependentRandomBrightnessContrast(brightness_limit=0.1,
                                                contrast_limit=0.1),
            A.RandomGamma(gamma_limit=(75, 125)),
            A.NoOp()
        ]),
        A.HorizontalFlip(p=0.5),
    ])

def get_medium_augmentations(image_size):
    return A.Compose([
        # A.Resize(int(image_size[0]/2), int(image_size[0]/2), p=1),
        # A.OneOf([
        #     # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
        #     #                    rotate_limit=0,
        #     #                    border_mode=cv2.BORDER_CONSTANT, value=0,p=0.5),
        #     A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                         value=0,p=0.5),
        #     A.NoOp()
        # ]),
        # A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.85), image_size[0]),
        #                   height=image_size[0],
        #                   width=image_size[1], p=0.3),
        # A.OneOf([
        #     A.RandomBrightnessContrast(brightness_limit=0.3,
        #                                contrast_limit=0.4),
        #     IndependentRandomBrightnessContrast(brightness_limit=0.25,
        #                                         contrast_limit=0.24),
        #     A.RandomGamma(gamma_limit=(50, 150)),
        #     A.NoOp()
        # ]),
        # A.OneOf([
        # #     A.MotionBlur(blur_limit=3),
        # #     A.GaussianBlur(),
        #     A.GaussNoise(),
        #     A.NoOp()
        # ]),
        # A.OneOf([
        #     A.RGBShift(r_shift_limit=5, b_shift_limit=5, g_shift_limit=5),
        #     A.HueSaturationValue(hue_shift_limit=5,
        #                          sat_shift_limit=5),
        #     A.NoOp()
        # ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])


def get_hard_augmentations(image_size):
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=0,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=0,
                               alpha=35,
                               sigma=5,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                             value=0),
            A.NoOp()
        ]),

        A.OneOf([
            A.ZeroTopAndBottom(p=0.3),

            A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                              height=image_size[0],
                              width=image_size[1], p=0.3),
            A.NoOp()
        ]),

        A.ISONoise(p=0.5),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),

        A.ChannelDropout(),
        A.RandomGridShuffle(p=0.3),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ])
    ])

def get_hard_augmentations_v2(image_size):
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=45,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=0,
                               alpha=35,
                               sigma=5,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                             value=0),
            A.NoOp()
        ]),

        A.OneOf([
            A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                              height=image_size[0],
                              width=image_size[1], p=0.3),
            A.NoOp()
        ]),

        A.ISONoise(p=0.5),
        A.JpegCompression(p=0.3, quality_lower=75),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),


        A.OneOf([
            A.ChannelDropout(p=0.2),
            A.CoarseDropout(p=0.1, max_holes=2, max_width=256, max_height=256, min_height=16, min_width=16),
            A.NoOp()
        ]),

        A.RandomGridShuffle(p=0.3),
        A.DiagnosisNoise(p=0.2),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ])
    ])