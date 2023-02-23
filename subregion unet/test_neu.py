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
from datagenerator_neu import DataGenerator

from torchvision.transforms import transforms
from model_p import fianlModel
from dataAug import Compose,ToTensor,TestRescale

def test(cfg):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed_all(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader


    n_classes = 4 # the number of defects
    test_transforms = Compose([TestRescale(input_hw=(192, 192)),
                               ToTensor(),  # /255
                               # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ])

    t_loader = DataGenerator(txtpath='/dataset/neutest.txt', transformer=test_transforms)
    valloader = data.DataLoader(
        t_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Modeld

    model = fianlModel(inchannel=3, nclass=n_classes).to(device)

    checkpoint = torch.load("./modelSavePath/finalModel_best_newmodel_neu.pkl")

    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    #compute the complexity of model
    alltime=0
    for i in range(100):

        img = torch.tensor(np.ones([1, 3, 192, 192]))
        img=img.to(device,dtype=torch.float)
    
        start_ts = time.time()
        result = model(img)
        end_ts = time.time()
        alltime += (end_ts - start_ts)
    print(alltime/100)
    # print(end_ts - start_ts)



    with torch.no_grad():

        for i_val, (images_val, labels_val,label1) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device, dtype=torch.float)
            # images_val = local_contrast_norm(images_val)
            labels_val = labels_val.squeeze(1).to(device, dtype=torch.int64)

            outputs = model(images_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()

            running_metrics_val.update(gt, pred)
            # val_loss_meter.update(val_loss.item())

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

    # run_id = random.randint(1, 100000)
    run_id = 100

    test(cfg)
