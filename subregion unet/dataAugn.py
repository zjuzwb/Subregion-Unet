import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf


def saltpepper(img, n):
    m = int((img.shape[0] * img.shape[1]) * n)
    for a in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    return img
class Augmentations_PIL:
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
        self.image_fill = 0  # image fill=0，0对应黑边
        self.label_fill = 0  # label fill=0，0对应黑边

    def perpernoise(self, image, label, label1):
        # assert len(kernel_size) == 2, "kernel size must be tuple and len()=2"
        n=random.choice([0.001, 0.01,0.0001])

        image = saltpepper(np.array(image), n)
        image = Image.fromarray(image, "RGB")

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)

        return image, label,label1
class Transforms_PIL(object):
    def __init__(self, input_hw=(192, 192)):
        self.aug_pil = Augmentations_PIL(input_hw)
        self.aug_funcs = [a for a in self.aug_pil.__dir__() if not a.startswith('_') and a not in self.aug_pil.__dict__]
        # print(self.aug_funcs)

    def __call__(self, image, label,label1):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :return:  PIL
        '''
        aug_name = random.choice(self.aug_funcs)
        # aug_name = 'random_resize_crop' #'random_rotate' #'random_flip' #'random_blur' #'random_noise' #'random_affine' #'random_resize_minify' #'random_resize_crop'
        # print(aug_name)  # 类实例后，读取数据时会不停的调用这个，每次都应该随机选择吧！
        image, label,label1 = getattr(self.aug_pil, aug_name)(image, label,label1)
        return image, label,label1

class TestRescale(object):
    # test
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
    def __call__(self, image, label,label1):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :return:  PIL
        '''
        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)
        return image, label,label1

class ToTensor(object):
    # image label -> tensor, image div 255
    def __call__(self, image, label,label1):
        # PIL uint8
        image = tf.to_tensor(image)  # transpose HWC->CHW, /255
        label = torch.from_numpy(np.array(label))  # PIL->ndarray->tensor
        label1 = torch.from_numpy(np.array(label1))
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        if not isinstance(label1, torch.LongTensor):
            label1 = label1.long()
        return image, label,label1

# class Normalize(object):
#     # (image-mean)/std
#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean  # RGB
#         self.std = std
#         self.inplace = inplace
#
#     def __call__(self, image, label):
#         image = tf.normalize(image, self.mean, self.std, self.inplace)
#         assert isinstance(label, torch.LongTensor)
#         label = label
#         return image, label

# Compose pytorch自带的只对img处理，需要重写
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label,label1):
        for t in self.transforms:
            image, label,label1 = t(image, label,label1)
        return image, label,label1

if __name__ == '__main__':

    image = np.uint8(np.random.rand(100,100,3)*255)
    label = np.ones([100,100], dtype=np.uint8)
    image = Image.fromarray(image, "RGB")  # PIL
    label = Image.fromarray(label)  # PIL
    # image1, label1 = trans(image, label)


    im_out, lab_out = train_transforms(image, label,label1)
    print(im_out.shape)