
from torch.utils.data import Dataset

import numpy as np
import cv2
import copy
from PIL import Image

class DataGenerator(Dataset):
    def  __init__(self, txtpath='/data/zwb/mld/', transformer = None):
        self.datatxtpath = txtpath
        self.transform = transformer
        self.mask, self.imgs, self.binarymask = self.read_txt()

    def read_txt(self):
        mask = []
        imgall = []
        binarymask = []
        with open(self.datatxtpath, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.strip(' ')
                img_mask_path = line.split(' ')
                # img = []
                for i in range(len(img_mask_path)):
                    img_mask = Image.open(img_mask_path[i])
                    # img_mask = Image.resize(img_mask,(192,192))


                    if i == 0:
                        img_mask0 = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
                        img_mask0 = self.encode_segmap(img_mask0)

                        mask0 = copy.deepcopy(img_mask0)

                        mask0[mask0[:, :] != 0] = 1
                        # mask0.astype(int)
                        mask0 = Image.fromarray(np.uint8(mask0))
                        binarymask.append(mask0)

                        # img_mask = img_mask.astype(int)
                        img_mask = Image.fromarray(np.uint8(img_mask0))

                        mask.append(img_mask)



                    else:
                        if len(img_mask.size)==3:

                            imgall.append(img_mask)

                        else:
                            img = img_mask.convert('RGB')

                            imgall.append(img)


        return mask, imgall, binarymask

    def __getitem__(self, index):
        mask = self.mask[index]
        # mask = self.encode_segmap(mask)
        img = self.imgs[index]
        binarymask = self.binarymask[index]
        img, mask, binarymask = self.transform(img, mask, binarymask)

        return img, mask, binarymask

    def __len__(self):
        return len(self.mask)





    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [255, 0, 0],#dent bgr
                [0, 255, 0],#deformation
                # [128, 128, 0],
                [0, 0, 255],#squlidity
                [128, 0, 128],#scratch
                [0, 128, 128],#spot

            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

