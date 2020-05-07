# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_train_loop.ipynb (unless otherwise specified).

__all__ = ['SEED', 'get_model', 'get_dataset', 'test_one_batch', 'batch_info', 'main_loop', 'evaluate_model']

# Cell

# general
import torch
from tqdm import tqdm, tqdm_notebook, trange

# import the relevant models
from .model.dcgan import DCGAN
from .model.diagnosis import TripleMRNet
from .model.vqvae import VQVAE2
from .model.introvae import IntroVAE
from .model.bigan import BiGAN
from .model.rnnvae import creator_rnn_ae
from .model.mocoae import MoCoAE

# import the dataset
from .dataloader import (
    load_test_batch,
    load_kneexray_datasets,
    load_mrnet_datasets,
)

# import the parameters
from .parameters import get_all_args, compat_args

# define the fixed randomness
SEED = 42

# Cell


def get_model(device, args):
    """
    return the required model depending on the arguments:
    """
    print(f"Model-Type: {args.model_type}")
    switcher = {
        "dcgan": DCGAN,
        "diagnosis": TripleMRNet,
        "vqvae": VQVAE2,
        "introvae": IntroVAE,
        "rnnvae": creator_rnn_ae,
        "bigan": BiGAN,
        "mocoae": MoCoAE,
    }
    # Get the model_creator
    model_creator = switcher.get(args.model_type, lambda: "Invalid Model Type")
    # create model
    return model_creator(device, args)

# Cell


def get_dataset(args):
    """
    return the required datasets and dataloaders depending on the dataset
    """
    print(f"Dataset-Type: {args.dataset_type}")
    switcher = {
        "MRNet": load_mrnet_datasets,
        "KneeXray": load_kneexray_datasets,
    }
    # Get the model_creator
    dataset_loader = switcher.get(args.dataset_type, lambda: "Invalid Model Type")
    # create model
    return dataset_loader(args)

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


def batch_info(test_x, args):
    """display some relevant infos about the dataset format and statistics"""

    if args.model_type == "diagnosis":
        print(test_x[args.perspectives[0]].shape)
        print(torch.min(test_x[args.perspectives[0]]))
        print(torch.max(test_x[args.perspectives[0]]))
    else:
        print(f"Input Dimension: {test_x.shape}")
        print(f"Minimum Value in Batch: {torch.min(test_x)}")
        print(f"Maximum Value in Batch: {torch.max(test_x)}")

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
    _, _, train_loader, valid_loader = get_dataset(args)

    # Initialise the Model
    model = get_model(device, args)

    # for reproducibility
    torch.manual_seed(SEED)

    # test dataset
    test_data = next(iter(train_loader))
    test_x = model.prep(test_data)

    batch_info(test_x, args)

    # Load pretrained params
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))

    # start training
    print("\nStart training")
    batch_count = 0
    model.train()

    for epoch in tq(range(args.n_epochs), desc="Epochs"):

        # remodify after pretraining training
        if epoch == args.n_pretrain and args.model_type in ("vqvae", "introvae"):
            print("parameters set")
            model.set_parameters(args)

        # For each batch in the dataloader
        for data in tq(train_loader, leave=False):
            # perform training
            model(data)

            # evaluate
            evaluate_model(args, model, batch_count, epoch, test_data, valid_loader)

            # watch progress
            batch_count += 1

# Cell

def evaluate_model(args, model, batch_count, epoch, test_data, valid_loader):
    """Evaluate the model every x iterations"""

    if batch_count % args.watch_batch == 0:
        # eval status
        model.eval()

        if args.model_type in ("triplenet", "diagnosis"):
            model.watch_progress(valid_loader, epoch)
        else:
            model.watch_progress(test_data, epoch)

        # reset to train status again
        model.train()