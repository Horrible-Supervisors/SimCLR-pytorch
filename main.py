import argparse
import os
import numpy as np
import torch
import torchvision
import time

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet, NT_Xent_With_Neg_Samples
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook, data, DataManipulator


def train(args, train_loader, model, criterion, optimizer, writer,
          neg_samples_loader, steps_per_epoch):

    if neg_samples_loader is not None:
        iter_obj = iter(neg_samples_loader)

    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        if step >= steps_per_epoch:
            break
        # print(f"step: {step}", flush=True)
        ns = None
        if neg_samples_loader is not None:
            # neg_samples_loader.dataset.randomize_samples()
            ns = next(iter_obj)[0]

        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        if neg_samples_loader is not None:
            ns = ns.cuda(non_blocking=True)
            h_i, h_j, h_ns, z_i, z_j, z_ns = model(x_i, x_j, ns)
            loss = criterion(z_i, z_j, z_ns)
        else:
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)

        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and (step % 50 == 0 or step == steps_per_epoch-1):
            print(f"Step [{step}/{steps_per_epoch}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch",
                              loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


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
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "Imagenet-1pct":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/imagenet-1pct/train.csv",
            args.dataset_dir + "/imagenet-1pct/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "Imagenet-10pct":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/imagenet-10pct/train.csv",
            args.dataset_dir + "/imagenet-10pct/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "HS-Imagenet":
        train_dataset = data.ImagenetDataset(
            args.train_csv,
            args.dataset_dir + "/imagenet/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "Demon-Imagenet":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/train-r.csv",
            args.dataset_dir + "/demon-dataset/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size,
            rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    train_steps = len(train_dataset) * args.epochs // args.batch_size + 1
    steps_per_epoch = int(train_steps / args.epochs)
    if args.include_neg_samples:
        train_steps = len(
            train_dataset
        ) * args.epochs // (args.batch_size + args.ns_batch_size/2) + 1
        steps_per_epoch = int(train_steps / args.epochs)

    neg_samples_loader = None
    if args.include_neg_samples:

        neg_sample_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                        size=args.image_size),
                    torchvision.transforms.CenterCrop(
                        size=args.image_size),
                    torchvision.transforms.ToTensor(),
                ])
        if args.dataset == "Imagenette":
            neg_samples_dataset = data.NegativeImagenetDataset(
                images_folder=args.dataset_dir + "/imagenette/negative_samples/",
                batch_size=args.ns_batch_size,
                n_img_class=args.n_img_class,
                n_img_samples_per_class=args.n_img_samples_per_class,
                class_remapping_file_path=None,
                epochs=args.epochs,
                train_steps=train_steps,
                steps_per_epoch=steps_per_epoch,
                transform=neg_sample_transform
            )
        elif args.dataset == "Demon-Imagenet":
            neg_samples_dataset = data.NegativeImagenetDataset(
                images_folder=args.dataset_dir + "/demon-dataset/negative_samples/",
                batch_size=args.ns_batch_size,
                n_img_class=args.n_img_class,
                n_img_samples_per_class=args.n_img_samples_per_class,
                class_remapping_file_path="./remappings/remapping-demon.pkl",
                epochs=args.epochs,
                train_steps=train_steps,
                steps_per_epoch=steps_per_epoch,
                transform= neg_sample_transform
            )
        neg_samples_loader = torch.utils.data.DataLoader(
            neg_samples_dataset,
            batch_size=neg_samples_dataset.batch_size,
            num_workers=args.workers
        )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(
            model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    if not args.include_neg_samples:
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    else:
        criterion = NT_Xent_With_Neg_Samples(
            args.batch_size, args.ns_batch_size,
            args.temperature, args.world_size
        )

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs+1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model,
                           criterion, optimizer, writer,
                           neg_samples_loader, steps_per_epoch)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)
        end = time.time()
        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch /
                              len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"""Epoch [{epoch}/{args.epochs}]\t Loss: {
                    loss_epoch / len(train_loader)}\t lr: {
                        round(lr, 5)}\nEpoch Time: {end - start:.2f} sec"""
            )
            args.current_epoch += 1

    # end training
    save_model(args, model, optimizer)


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
