import cv2
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
'''
    https://pytorch.org/docs/1.6.0/torchvision/transforms.html#functional-transforms
'''
class Augmentations_PIL:
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
        self.image_fill = 0  # image fill=0，0对应黑边
        self.label_fill = 0  # label fill=0，0对应黑边
    def random_rotate(self, image, label, label1, angle=[90, 180, 270]):
        if angle is None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        elif isinstance(angle, list) or isinstance(angle, tuple):
            angle = random.choice(angle)

        image = tf.rotate(image, angle)
        label = tf.rotate(label, angle)
        label1 = tf.rotate(label1, angle)

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)

        return image, label,label1

    def random_flip(self, image, label,label1):
        if random.random() > 0.5:
            image = tf.hflip(image)
            label = tf.hflip(label)
            label1 = tf.hflip(label1)
        if random.random() < 0.5:
            image = tf.vflip(image)
            label = tf.vflip(label)
            label1 = tf.vflip(label1)

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)

        return image, label,label1

    def random_noise(self, image, label,label1, noise_sigma=10):
        in_hw = image.size[::-1] + (1,)
        noise = np.uint8(np.random.randn(*in_hw) * noise_sigma)  # +-

        image = np.array(image) + noise  # broadcast
        image = Image.fromarray(image, "RGB")

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)

        return image, label,label1

    def random_blur(self, image, label, label1, kernel_size=(3,3)):
        assert len(kernel_size) == 2, "kernel size must be tuple and len()=2"
        image = cv2.GaussianBlur(np.array(image), ksize=kernel_size, sigmaX=0)
        image = Image.fromarray(image, "RGB")

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)
        label1 = tf.resize(label1, self.input_hw, interpolation=Image.NEAREST)

        return image, label,label1

class Transforms_PIL(object):
    def __init__(self, input_hw=(256, 256)):
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


    im_out, lab_out = train_transforms(image, label,label1)
    print(im_out.shape)