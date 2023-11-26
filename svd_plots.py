import os, pdb
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from utils import yaml_config_hook, data


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def main(args):
    train_dataset = data.ImagenetDataset(
        args.dataset_dir + "/imagenette/train.csv",
        args.dataset_dir + "/imagenette/train",
        num_variations=0,
        transform_type=4,
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )
    test_dataset = data.ImagenetDataset(
        args.dataset_dir + "/imagenette/val.csv",
        args.dataset_dir + "/imagenette/val",
        num_variations=0,
        transform_type=4,
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)

    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    W = simclr_model.projector[0].weight.cpu().detach().numpy()
    svd_out = np.linalg.svd(W)
    eigenvalues = svd_out.S
    plt.plot(eigenvalues)
    plt.savefig("svd_0.png")
    plt.close()

    plt.plot(np.log10(eigenvalues))
    plt.savefig("svd_log_0.png")
    plt.close()


if __name__ == '__main__':

    config_parser = argparse.ArgumentParser(description="Config")
    config_parser.add_argument('--config', '-c', required=False, default="./config/config.yaml",
                               help="The config.yaml file to use. Contains the arguments for the training run.")
    config_args, _ = config_parser.parse_known_args()
    config_filepath = config_args.config

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(config_filepath)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)