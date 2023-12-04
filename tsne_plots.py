import os, pdb, json
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

from sklearn.manifold import TSNE

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from utils import yaml_config_hook, data


def inference(loader, simclr_model, device):
    feature_vector = []
    projection_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        projection_vector.extend(z.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    projection_vector = np.array(projection_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, projection_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_proj, train_y = inference(train_loader, simclr_model, device)
    test_X, test_proj, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_proj, train_y, test_X, test_proj, test_y

def get_features_train(simclr_model, train_loader, device):
    train_X, train_proj, train_y = inference(train_loader, simclr_model, device)
    return train_X, train_proj, train_y

def get_features_val(simclr_model, test_loader, device):
    test_X, test_proj, test_y = inference(test_loader, simclr_model, device)
    return test_X, test_proj, test_y

def main(args):

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
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/train-r.csv",
            args.dataset_dir + "/demon-dataset/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
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
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/almighty-dataset/val-r.csv",
            args.dataset_dir + "/almighty-dataset/val",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "Dogs-Imagenet":
        train_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/train-50-1000-r-dogs.csv",
            args.dataset_dir + "/demon-dataset/train",
            num_variations=args.num_variations,
            transform_type=args.transform_type,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        val_dataset = data.ImagenetDataset(
            args.dataset_dir + "/demon-dataset/val-50-1000-r-dogs.csv",
            args.dataset_dir + "/demon-dataset/val",
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

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(
        args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(
        model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    print("### Creating features from pre-trained context model ###")
    test_X, test_proj, test_y = get_features_val(simclr_model, val_loader, args.device)
    # (train_X, train_proj, train_y, test_X, test_proj, test_y) = get_features(
    #     simclr_model, train_loader, val_loader, args.device
    # )

    with open("imagenet_class_index.json") as inp_fle:
        class_data_dict = json.load(inp_fle)

    df = val_dataset.image_frame
    mapping = df.loc[~(df['label'].duplicated().values), ['label', 'original_label']]
    map_dict = {}
    other_map_dict = {}
    for idx, row in mapping.iterrows():
        other_map_dict[str(row['label'])] = str(row['original_label'])
        map_dict[str(row['label'])] = class_data_dict[str(row['original_label'])][1]

    out_data = {'test_X': test_X, 'test_proj': test_proj, 'test_y': test_y}
    with open("tsne_data.pkl", "wb") as out_fle:
        pkl.dump(out_data, out_fle)

    # train_df = train_dataset.image_frame
    # mask_stuff = train_df.apply(lambda x: str(x['original_label']) in other_map_dict.values(), axis=1)
    # dogs_train_df = train_df.loc[mask_stuff.values]

    orig_class_labels = np.array([158,161,205,206,211,226,228,238,239,247,258,270,276,304,307,310,311,319,325,847,857,867])

    train_df = train_dataset.image_frame
    train_df = train_df.loc[train_df['original_label'].isin(orig_class_labels).values]
    # df = df.loc[df['original_label'].isin(orig_class_labels).values]
    pdb.set_trace()

    class_labels = [str(x) for x in np.unique(df['label'])]
    class_labels= np.random.choice(class_labels, size=min(10, len(class_labels)), replace=False)

    mask = []
    for x in test_y:
        if str(x) in class_labels:
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    remapped_test_y = np.array([map_dict[str(y)] for y in test_y])
    remapped_class_labels = np.array([map_dict[str(y)] for y in class_labels])

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(test_X[mask])
    # X_tsne = tsne.transform(test_X)

    scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=test_y[mask])
    handles, _ = scatter.legend_elements(prop='colors', num=None)
    # plt.legend(handles, remapped_class_labels, bbox_to_anchor=(1.05, 1.05))

    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # plt.gca().set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])

    # Put a legend to the right of the current axis
    # plt.gca().legend(handles, remapped_class_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.gca().legend(handles, remapped_class_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, ncols=1, fontsize=6)


    # plt.gca().legend(handles, remapped_class_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.suptitle("T-SNE 10 Randomly Selected", fontsize=12)
    name_list = os.path.basename(args.model_path).split('-')
    out_name = '-'.join([name_list[1], name_list[2], name_list[4], 'tsne.png'])
    plt.savefig(out_name)
    # pdb.set_trace()


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