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

def load_model(model_name):
    if model_name == "ResNet18":
        return torchvision.models.resnet18()
    else:
        return torchvision.models.resnet50()
    
def load_ds(path_to_train, path_to_valid):
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=45),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=path_to_train, 
        transform=transform
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root=path_to_valid,
        transform=transform
    )
    print("Datasets successfully loaded")
    return train_dataset, valid_dataset

def train(args, train_loader, model, criterion, optimizer, writer,
          neg_samples_loader, steps_per_epoch):

    loss_epoch = 0
    for step, (x_i, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        pred = model(x_i)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and (step % 50 == 0 or step == steps_per_epoch-1):
            print(f"Step [{step}/{steps_per_epoch}]\t Loss: {loss.item()}", flush=True)

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch",
                              loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch

def train_model(args, model, train_loader, val_loader, steps_per_epoch):
    model = model.to(args.device)
    print("Training on " + args.device.type)

    neg_samples_loader = None

    writer = None
    if args.nr == 0:
        writer = SummaryWriter(log_dir=args.model_path)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer, scheduler = load_optimizer(args, model)

    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs+1):
        start = time.time()
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

        # valid_loss = valid_model(args, model, val_loader)
        # print(f'Epoch {epoch+1}/{args.epochs}, Validation Loss: {valid_loss:.4f}')

    # end training
    save_model(args, model, optimizer)

    # for epoch in range(num_epochs):
    #     print(f'Begin training Epoch {epoch+1}/{num_epochs}')
    #     acc_loss = 0
    #     for inputs, labels in train:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         acc_loss += loss
    #         loss.backward()
    #         optimizer.step()

    #     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {acc_loss:.4f}')

    #     valid_loss = valid_model(model, valid)
    #     print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss:.4f}')

    # # save the model parameters
    # torch.save(model.state_dict(), str(num_epochs) +  '_' + str(batch_size) + '_' + str(learning_rate) + '.pth')

def valid_model(args, model, valid):
    acc_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for inputs, labels in valid:
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc_loss += loss
    
    return acc_loss

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
        if args.transform_type == 5:
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/demon-dataset/train-r.csv",
                args.dataset_dir + "/demon-dataset/train",
                num_variations=args.num_variations,
                transform_type=args.transform_type,
                transform=TransformsSimCLR(size=args.image_size).variation_transform,
            )
        else:
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/demon-dataset/train-r.csv",
                args.dataset_dir + "/demon-dataset/train",
                num_variations=args.num_variations,
                transform_type=args.transform_type,
                transform=TransformsSimCLR(size=args.image_size).train_transform,
            )
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/val-r.csv",
            args.dataset_dir + "/demon-dataset/val",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "Almighty-Imagenet":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/almighty-dataset/train-r.csv",
            args.dataset_dir + "/almighty-dataset/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).train_transform,
        )
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/almighty-dataset/val-r.csv",
            args.dataset_dir + "/almighty-dataset/val",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    train_steps = len(train_dataset) * args.epochs // args.batch_size + 1
    steps_per_epoch = int(train_steps / args.epochs)

    model = get_resnet(args.resnet, pretrained=False)
    model.fc.out_features = args.n_classes

    train_model(args, model, train_loader, val_loader, steps_per_epoch)

    # model = load_model(model_name)
    # train_ds, valid_ds = load_ds("/home/jrick6/pytorch_datasets/almighty-dataset/train", "/home/jrick6/pytorch_datasets/almighty-dataset/val")
    # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    # valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    # train_model(model, train_loader, valid_loader)


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

    print("Dataset: ", args.dataset)
    print("Transform type: ", args.transform_type)
    print("Batch size: ", args.batch_size)
    print("Number of epochs: ", args.epochs)
    print("ResNet: ", args.resnet)
    print("Model Path: ", args.model_path)
    print("Include negative samples: ", args.include_neg_samples)
    print("Number of negative samples: ", args.n_img_samples_per_class)

    print(f"Total time: {t_end - t_start} seconds")