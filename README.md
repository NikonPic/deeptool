# deeptool
> Tool for applying several deep learning methods on the MRNet dataset.



<img src="nbs\img\deeptool.png" alt="Drawing" style="width: 500px;">

Current implementations:

* DIAGNOSIS (MRNet based)

* DCGAN
* VQVAE - VQVAE2
* INTROVAE



## Install

Later this will simply be something like:

`pip install deeptool`

For now the source code needs to be downloaded.
Then navigation to the downloaded folder is required and the package can be installed with:

`pip install -e .`

## How to use

### 1. Import the library and some functionality

```python
from deeptool.train_loop import get_model, test_one_batch, main_loop
from deeptool.parameters import get_all_args, compat_args
```

### 2. Adjust the arguments for training for your needs
Example: (check parameters for more info)

```python
# Define Arguments (Example)
args = get_all_args()
args.dim = 2 # Dimension of network reduced to 2
args.batch_size = 1
args.perspectives = ["coronal"] # View only the coronal planes
args.n_res_2d = 1 # architecture contains resnet blocks
args.watch_batch = 100 # visualize every 100 batches
args.model_type = "introvae" # train the introvae model
args = compat_args(args)
```

### 3. Train the model

```python
main_loop(args) # run the training
```
