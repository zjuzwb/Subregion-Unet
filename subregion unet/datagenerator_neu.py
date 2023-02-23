
from torch.utils.data import Dataset
import numpy as np
import copy
import matplotlib.pyplot as plt

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

                    if i == 0:
                        if len(img_mask.size) == 3:

                            img_mask = img_mask.convert('L')#cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
                            mask0 = copy.deepcopy(img_mask)
                            # edgemask0 = self.edgemaskp()
                            mask0 = np.array(mask0)
                            mask0[mask0[:,:]!=0] = 1
                            mask0=Image.fromarray(mask0)
                            binarymask.append(mask0)

                            mask.append(img_mask)
                        else:
                            mask0 = copy.deepcopy(img_mask)
                            mask0 = np.array(mask0)
                            mask0[mask0[:, :] != 0] = 1
                            mask0=Image.fromarray(mask0)
                            binarymask.append(mask0)
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
        img = self.imgs[index]
        binarymask = self.binarymask[index]
        img, mask,binarymask = self.transform(img, mask,binarymask)

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
                [255, 0, 0],
                [0, 255, 0],
                # [128, 128, 0],
                [0, 0, 255],

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

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

