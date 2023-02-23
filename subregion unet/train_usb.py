
import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm

from tools.loss import get_loss_function
from tools.loader import get_loader
from tools.utils import get_logger
from tools.computemIou import runningScore, averageMeter
from tools.schedulers import get_scheduler
from tools.optimizers import get_optimizer

from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
import torch.nn
from dataAug import Compose, Transforms_PIL,ToTensor,TestRescale

from datagenerator_usb import DataGenerator
from model_p import fianlModel#, localContextNet

def train(cfg, writer, logger):

    # viz = Visdom(   )
    # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    train_transforms = Compose([Transforms_PIL(input_hw=(192, 192)),
                                ToTensor(),])

    test_transforms = Compose([TestRescale(input_hw=(192, 192)),
                               ToTensor(),  # /255
                               # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ])
    train_loader = DataGenerator(txtpath='./dataset/usbtrain.txt', transformer=train_transforms, mode='train')

    n_classes = 6
    trainloader = data.DataLoader(
        train_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        drop_last=True
    )

    # 1tf =



    t1_loader = DataGenerator(txtpath='/data/zwb/test_no_reference/usbtrain.txt', transformer=test_transforms, mode='test')
    # t1_loader = DataGenerator(txtpath='/data/zwb/Test_isuseful/train.txt', transformer=tf00, mode='test')

    valloader = data.DataLoader(
        t1_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"], drop_last=True
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = fianlModel(n_classes=n_classes).to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels,label1) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device, dtype=torch.float)

            # lcn_norm
            # images = local_contrast_norm(images, 7)
            labels = labels.squeeze(1).to(device, dtype=torch.int64)
            # label1 = label1.squeeze(1).to(device, dtype=torch.int64)

            optimizer.zero_grad()
            outputs= model(images)

            loss = loss_fn(input=outputs, target=labels)
            # loss1 = loss_fn(maskpre, label1)
            # loss = loss#+10*loss1# 0.7991744037994424

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)
            # viz.line([loss.item()], [i], win='trian_loss', update='append')

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val,label1) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device, dtype=torch.float)
                        # lcn_norm
                        # images_val = local_contrast_norm(images_val, 7)

                        labels_val = labels_val.squeeze(1).to(device, dtype=torch.int64)
                        # label1 = label1.squeeze(1).to(device, dtype=torch.int64)

                        outputs= model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)
                        # loss1 = loss_fn(maskout, label1)
                        # val_loss = val_loss +10* loss1

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        'model_savePath',
                                "{}_{}_best_newmodel_usb.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


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
    run_id='all_img'
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
