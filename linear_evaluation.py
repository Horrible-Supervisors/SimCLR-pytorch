import os
import argparse
import torch
import torchvision
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook, data, visualizations


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


def create_data_loaders_from_arrays(X_train, y_train,
                                    X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for _, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    accuracy3_epoch = 0
    accuracy5_epoch = 0
    model.eval()

    actual_class_ids = []
    predicted_class_ids = []

    for _, (x, y) in enumerate(loader):
        actual_class_ids = actual_class_ids + y.tolist()
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        predicted_class_ids = predicted_class_ids + predicted.tolist()

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

    return (loss_epoch, accuracy_epoch, accuracy3_epoch, accuracy5_epoch,
            actual_class_ids, predicted_class_ids)


if __name__ == "__main__":

    config_parser = argparse.ArgumentParser(description="Config")
    config_parser.add_argument(
        '--config', '-c', required=False, default="./config/config.yaml",
        help="""The config.yaml file to use. """
             """Contains the arguments for the training run.""")
    config_parser.add_argument(
        '--use-pets', '-p', action='store_true', required=False,
        help="""Whether to use pets dataset.""")
    config_parser.add_argument(
        '--use-caltech', '-ct', action='store_true', required=False,
        help="""Whether to use pets dataset.""")
    config_parser.add_argument(
        '--use-dogs', '-d', action='store_true', required=False,
        help="""Whether to use only dogs from pets dataset.""")
    config_args, _ = config_parser.parse_known_args()
    config_filepath = config_args.config

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(config_filepath)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args, _ = parser.parse_known_args()

    if args.dataset == "HS-Imagenet":
        train_csv = f"/imagenet/train-{args.n_classes}-{args.n_img_class}.csv"
        args.train_csv = args.dataset_dir + train_csv
        val_csv = f"/imagenet/val-{args.n_classes}-{args.n_img_class}.csv"
        args.val_csv = args.dataset_dir + val_csv

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_args.use_pets:
        if config_args.use_dogs:
            train_dataset = data.PetsDataset(
                args.dataset_dir+'/pets', train=True, dogs=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform
            )
            test_dataset = data.PetsDataset(
                args.dataset_dir+'/pets', train=False, dogs=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform
            )
        else:
            train_dataset = data.PetsDataset(
                args.dataset_dir+'/pets', train=True, dogs=False,
                transform=TransformsSimCLR(size=args.image_size).test_transform
            )
            test_dataset = data.PetsDataset(
                args.dataset_dir+'/pets', train=False, dogs=False,
                transform=TransformsSimCLR(size=args.image_size).test_transform
            )
    elif config_args.use_caltech:
        train_dataset = data.CaltechDataset(
            args.dataset_dir+'/caltech-101', train=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform
        )
        test_dataset = data.CaltechDataset(
            args.dataset_dir+'/caltech-101', train=False,
            transform=TransformsSimCLR(size=args.image_size).test_transform
        )
    else:
        if args.dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="test",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=False,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "Imagenette":
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
        elif args.dataset == "Imagenet-1pct":
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/imagenet-1pct/train.csv",
                args.dataset_dir + "/imagenet-1pct/train",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = data.ImagenetDataset(
                args.dataset_dir + "/imagenet-val/val.csv",
                args.dataset_dir + "/imagenet-val/val",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "Imagenet-10pct":
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/imagenet-10pct/train.csv",
                args.dataset_dir + "/imagenet-10pct/train",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = data.ImagenetDataset(
                args.dataset_dir + "/imagenet-val/val.csv",
                args.dataset_dir + "/imagenet-val/val",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "HS-Imagenet":
            train_dataset = data.ImagenetDataset(
                args.train_csv,
                args.dataset_dir + "/imagenet/train",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = data.ImagenetDataset(
                args.val_csv,
                args.dataset_dir + "/imagenet/val",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "Demon-Imagenet":
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/demon-dataset/train-r.csv",
                args.dataset_dir + "/demon-dataset/train",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = data.ImagenetDataset(
                args.dataset_dir + "/demon-dataset/val-r.csv",
                args.dataset_dir + "/imagenet/val",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.dataset == "Almighty-Imagenet":
            train_dataset = data.ImagenetDataset(
                args.dataset_dir + "/almighty-dataset/train-r.csv",
                args.dataset_dir + "/almighty-dataset/train",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = data.ImagenetDataset(
                args.dataset_dir + "/almighty-dataset/val-r.csv",
                args.dataset_dir + "/imagenet/val",
                num_variations=0,
                transform_type=4,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        else:
            raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
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
    model_fp = os.path.join(
        args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(
        model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    # Logistic Regression
    n_classes = args.n_classes
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, model, criterion, optimizer
        )
        print(f"""Epoch [{epoch}/{args.logistic_epochs}]\t """
              f"""Loss: {loss_epoch / len(arr_train_loader)}\t """
              f"""Accuracy: {accuracy_epoch / len(arr_train_loader)}""")
        if epoch % 50 == 0:
            (loss_epoch, accuracy_epoch, accuracy3_epoch,
             accuracy5_epoch, _, _) = test(
                args, arr_test_loader, model, criterion)
            print(f"""[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t """
                  f"""Accuracy: {accuracy_epoch / len(arr_test_loader)}\t """
                  f"""Accuracy Top 3: {
                      accuracy5_epoch / len(arr_test_loader)}\t """
                  f"""Accuracy Top 5: {
                      accuracy5_epoch / len(arr_test_loader)}""")

    # final testing
    (loss_epoch, accuracy_epoch, accuracy3_epoch, accuracy5_epoch,
     actual_class_ids, predicted_class_ids) = test(
        args, arr_test_loader, model, criterion)
    print(f"""[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t """
          f"""Accuracy: {accuracy_epoch / len(arr_test_loader)}\t """
          f"""Accuracy Top 3: {accuracy3_epoch / len(arr_test_loader)}\t """
          f"""Accuracy Top 5: {accuracy5_epoch / len(arr_test_loader)}""")

    print("Dataset: ", args.dataset)
    print("Transform type: ", args.transform_type)
    print("Batch size: ", args.batch_size)
    print("Number of epochs: ", args.epoch_num)
    print("ResNet: ", args.resnet)
    print("Model Path: ", args.model_path)
    print("Include negative samples: ", args.include_neg_samples)
    print("Number of negative samples: ", args.n_img_samples_per_class)
    if config_args.use_pets:
        print("Use pets dataset: ", config_args.use_pets)
        print("Use dogs from pets dataset: ", config_args.use_dogs)
    elif config_args.use_caltech:
        print("Use caltech dataset: ", config_args.use_caltech)

    visualizations.plot_confusion_matrix(
        actual_class_ids, predicted_class_ids, 'Actual Class Ids',
        'Predicted Class Ids', 'CF_Grid_' + args.model_path + '_' +
        str(args.transform_type) + '_' + str(args.include_neg_samples) +
        '.png', args.model_path + '_' + str(args.transform_type) + '_' +
        str(args.include_neg_samples), show_grid=True, unique_id_count=50
    )
    visualizations.plot_confusion_matrix(
        actual_class_ids, predicted_class_ids, 'Actual Class Ids',
        'Predicted Class Ids', 'CF_' + args.model_path + '_' +
        str(args.transform_type) + '_' + str(args.include_neg_samples) +
        '.png', args.model_path + '_' + str(args.transform_type) + '_' +
        str(args.include_neg_samples), show_grid=False, unique_id_count=50
    )
