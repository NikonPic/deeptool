# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\04_train_loop.ipynb (unless otherwise specified).

__all__ = ['get_model', 'test_one_batch', 'main_loop']

# Cell

# general
import torch
from tqdm import tqdm, tqdm_notebook

# import the relevant models
from .model.dcgan import DCGAN
from .model.diagnosis import TripleMRNet
from .model.vqvae import VQVAE2
from .model.introvae import IntroVAE
from .model.bigan import BiGAN
from .model.rnnvae import Creator_RNN_AE
from .model.mocoae import MoCoAE

# import the dataset
from .dataloader import load_datasets, load_test_batch

# import the parameters
from .parameters import get_all_args, compat_args

# Cell


def get_model(device, args):
    """
    return the required model depending on the arguments:
    """
    print(args.model_type)
    switcher = {
        "dcgan": DCGAN,
        "diagnosis": TripleMRNet,
        "vqvae": VQVAE2,
        "introvae": IntroVAE,
        "rnnvae": Creator_RNN_AE,
        "bigan": BiGAN,
        "mocoae": MoCoAE,
    }
    # Get the model_creator
    model_creator = switcher.get(args.model_type, lambda: "Invalid Model Type")
    # create model
    return model_creator(device, args)

# Cell


def test_one_batch(args):
    """
    Useful functionality to test a new model using a demo databatch and check compatibility
    """
    args.track = False
    # define calculating device
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu"
    )
    # avoid dataloaders for testing on server:
    batch = load_test_batch(args)
    # Initialise the Model
    model = get_model(device, args)
    # switch to train mode
    model.train()
    # evaluate batch
    model(batch)

# Cell


def main_loop(args, tq_nb=True):
    """Perform the Training using the predefined arguments"""

    # select tqdm_type
    tq = tqdm
    if tq_nb:
        tq = tqdm_notebook

    # define calculating device
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu"
    )

    # build the dataloaders
    train_data, _, train_loader, valid_loader = load_datasets(args)

    # Initialise the Model
    model = get_model(device, args)

    # Load pretrained params
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))

    # test dataset
    test_data = next(iter(valid_loader))
    print("Input dimension:")

    if args.model_type == "diagnosis":
        print(test_data["img"][args.perspectives[0]].shape)
        print(torch.min(test_data["img"][args.perspectives[0]]))
        print(torch.max(test_data["img"][args.perspectives[0]]))
    else:
        print(test_data["img"].shape)
        print(torch.min(test_data["img"]))
        print(torch.max(test_data["img"]))

    # start training
    print("Start training")
    batch_count = 0
    model.train()

    for epoch in tq(range(args.n_epochs)):

        # remodify after pretraining training
        if epoch == args.n_pretrain and args.model_type in ("vqvae", "introvae"):
            print("parameters set")
            model.set_parameters(args)

        # For each batch in the dataloader
        for data in tq(train_loader):
            # perform training
            model(data)

            if batch_count % args.watch_batch == 0:
                # eval status
                model.eval()

                if args.model_type in ("triplenet", "diagnosis"):
                    model.watch_progress(valid_loader, epoch)
                else:
                    model.watch_progress(test_data, epoch)

                # reset to train status again
                model.train()

            # watch progress
            batch_count += 1