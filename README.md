# deeptool
> Tool for applying several deep learning methods on the MRNet and KneeXray dataset.



<img src="nbs\img\deeptool.png" alt="Drawing" style="width: 500px;">

Current implementations:

* DIAGNOSIS (MRNet based)

* DCGAN
* VQVAE - VQVAE2
* INTROVAE
* BIGAN

Own Creations:
* MoCoAE, MocoGAN
* Transition Networks for pseudo 3D analysis for DCGAN, INTROVAE, BIGAN 



## Install

Later this will simply be something like:

`pip install deeptool`

For now the source code needs to be downloaded.
Then navigation to the downloaded folder is required and the package can be installed with:

`pip install -e .`

## How to use

### 1. Import the library

```
from deeptool.train_loop import get_model, test_one_batch, main_loop
from deeptool.parameters import get_all_args, compat_args
```

### 2. Adjust the arguments for training for your needs
check parameters.py for more information about the effect of each parameter

```
# Define Arguments (Example)
args = get_all_args()
args.dim = 2  # Dimension of network reduced to 2
args.batch_size = 1 # How many pictures are included per training update
args.n_res_2d = 1  # architecture contains resnet blocks
args.watch_batch = 100  # visualize every 100 batches
args.model_type = "introvae"  # train the introvae model
args = compat_args(args) # solve argument interactions
```

### 3. Train the model

```python
main_loop(args) # run the training
```

### 4. Experience Results

Watch the model improve on the task, as displayed below...

<img src="nbs\img\movie.gif" alt="Drawing" style="width: 500px;">
