import os, pdb, time, argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from simclr.modules.transformations import TransformsSimCLR
from simclr.modules import get_resnet
from model import load_optimizer, save_model
from utils import yaml_config_hook, data, DataManipulator

def eval_model(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    accuracy3_epoch = 0
    accuracy5_epoch = 0
    model.eval()
    for _, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        predicted_top3 = output.topk(3, 1)[1]
        predicted_top5 = output.topk(5, 1)[1]
        acc = (predicted == y).sum().item() / y.size(0)

        acc_3 = sum([1 if y[i] in predicted_top3[i]
                    else 0 for i in range(len(y))]) / y.size(0)

        acc_5 = sum([1 if y[i] in predicted_top5[i]
                    else 0 for i in range(len(y))]) / y.size(0)
        accuracy_epoch += acc
        accuracy3_epoch += acc_3
        accuracy5_epoch += acc_5

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch, accuracy3_epoch, accuracy5_epoch

def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "Imagenette":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/imagenette/train.csv",
            args.dataset_dir + "/imagenette/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).train_transform,
        )
    elif args.dataset == "Imagenet-1pct":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/imagenet-1pct/train.csv",
            args.dataset_dir + "/imagenet-1pct/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).train_transform,
        )
    elif args.dataset == "Imagenet-10pct":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/imagenet-10pct/train.csv",
            args.dataset_dir + "/imagenet-10pct/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).train_transform,
        )
    elif args.dataset == "HS-Imagenet":
        train_dataset = data.ImagenetDataset(
            args.train_csv,
            args.dataset_dir + "/imagenet/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).train_transform,
        )
    elif args.dataset == "Demon-Imagenet":
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/val-r.csv",
            args.dataset_dir + "/demon-dataset/val",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "Almighty-Imagenet":
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/almighty-dataset/val-r.csv",
            args.dataset_dir + "/almighty-dataset/val",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # train_steps = len(train_dataset) * args.epochs // args.batch_size + 1
    # steps_per_epoch = int(train_steps / args.epochs)

    criterion = torch.nn.CrossEntropyLoss()

    model = get_resnet(args.resnet, pretrained=False)
    model.fc.out_features = args.n_classes

    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()

    loss_epoch, accuracy_epoch, accuracy3_epoch, accuracy5_epoch = eval_model(args, val_loader, model, criterion)
    print(f"""[FINAL]\t Loss: {loss_epoch / len(val_loader)}\t """
            f"""Accuracy: {accuracy_epoch / len(val_loader)}\t """
            f"""Accuracy Top 3: {accuracy3_epoch / len(val_loader)}\t """
            f"""Accuracy Top 5: {
                accuracy5_epoch / len(val_loader)}""")

if __name__ == "__main__":

    config_parser = argparse.ArgumentParser(description="Config")
    config_parser.add_argument(
        '--config', '-c', required=False, default="./config/config.yaml",
        help="""The config.yaml file to use. """
             """Contains the arguments for the training run.""")
    config_args, _ = config_parser.parse_known_args()
    config_filepath = config_args.config

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(config_filepath)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args, _ = parser.parse_known_args()

    print("Dataset: ", args.dataset)
    print("Transform type: ", args.transform_type)
    print("Batch size: ", args.batch_size)
    print("Number of epochs: ", args.epochs)
    print("ResNet: ", args.resnet)
    print("Model Path: ", args.model_path)

    t_start = time.time()

    if args.dataset == "HS-Imagenet":
        train_csv = f"/imagenet/train-{args.n_classes}-{args.n_img_class}.csv"
        args.train_csv = args.dataset_dir + train_csv
        val_csv = f"/imagenet/val-{args.n_classes}-{args.n_img_class}.csv"
        args.val_csv = args.dataset_dir + val_csv

        if not os.path.exists(train_csv):
            manipulator = DataManipulator(
                args.dataset_dir + "/imagenet/train.csv",
                args.dataset_dir + "/imagenet/val.csv",
                args.n_classes, args.n_img_class,
                args.dataset_dir + "/imagenet/")
            manipulator.create_csv(args.train_csv, args.val_csv, args.seed)

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8100"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"""Training with {args.nodes} nodes, waiting until
            all nodes join before starting training"""
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)

    t_end = time.time()

    print(f"Total time: {t_end - t_start} seconds")