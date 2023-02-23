import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

from tools.computemIou import runningScore, averageMeter

from datagenerator_usb import DataGenerator
from torchvision.transforms import transforms
from model_p import fianlModel
from dataAug import Compose,ToTensor,TestRescale

from thop import profile

# get labels
def get_pascal_labels():
    return np.asarray(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            # [128, 128, 0],
            [0, 0, 255],

        ]
    )


def encode_segmap( mask):
    mask = mask
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask):
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 4):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb


def train(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 6

    test_transforms = Compose([TestRescale(input_hw=(192, 192)),
                               ToTensor(),  # /255
                               # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ])
    t_loader = DataGenerator(txtpath='./dataset/usbtest.txt', transformer=test_transforms)

    valloader = data.DataLoader(
        t_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Modeld
    model = fianlModel(in_channels=3,n_classes=n_classes).to(device)



    checkpoint = torch.load("./modelSavePath/finalModel_best_newmodel_usb.pkl")
    model.load_state_dict(checkpoint["model_state"])
    val_loss_meter = averageMeter()

    model.eval()

   # compute the complexity of model
    timeall=0
    for gggg in range(100):
        input = torch.randn(1, 3, 192, 192)
        inputs = input.to(device)
        torch.cuda.synchronize()
        start = time.time()
        result = model(inputs)
        result = result.data.max(1)[1].cpu().numpy()
        result = result.squeeze(0)
        img =decode_segmap(result)
        torch.cuda.synchronize()

        end = time.time()
        a = end - start
        timeall = timeall + a
        # print(a)
    print(timeall/100)
    input = torch.randn(1, 3, 192, 192)
    input = input.to(device, dtype=torch.float)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    input = torch.randn(1, 3, 192, 192)
    input = input.to(device, dtype=torch.float)
    flops, params = profile(model, inputs=(input,))
    # test
    with torch.no_grad():

        for i_val, (images_val, labels_val,label1) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device, dtype=torch.float)
            # images_val = local_contrast_norm(images_val)
            labels_val = labels_val.squeeze(1).to(device, dtype=torch.int64)


            outputs = model(images_val)
            # val_loss = loss_fn(input=outputs, target=labels_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)



    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics_val.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="unet_pascal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)


    test(cfg)
