# %%
# hide
from nbdev.export import notebook2script

# Apply latest updates on the python files
notebook2script()
%matplotlib qt

# %%
from deeptool.train_loop import get_model, test_one_batch, main_loop
from deeptool.parameters import get_all_args, compat_args

# import the dataset
from deeptool.dataloader import (
    load_test_batch,
    load_kneexray_datasets,
    load_mrnet_datasets,
)

args = get_all_args()
args.dataset_type = "MRNet"
args.dim = 3
args.n_res2d = 1
args.evo_on = True
args.batch_size = 16
args.perspectives = ["coronal"]
args.watch_batch = 100
args.pic_size = 32

args.n_fea_down = 16
args.n_fea_up = 16

args.model_type = "bigan"
args.moco_aemode = True
args.moco_ganmode = True
args.n_epochs = 1000
args.moco_K = 500
args = compat_args(args)

_, _, train_loader, valid_loader = load_mrnet_datasets(args)


# %%
data = next(iter(train_loader))
print(data['img'].shape)
try:
    mri = data['img']['coronal'][0, :, :, :].numpy()
except:
    mri = data["img"][0, 0, :, :, :]


# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %%

xx, yy = np.meshgrid(np.linspace(0,1,args.pic_size), np.linspace(0,1,args.pic_size))
Z = np.ones(xx.shape)
data = mri[0, :, :]

# create the figure
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

for i in range(16):
    data = mri[i, :, :]
    ax2.plot_surface(xx, yy, Z * 2 *i, rstride=1, cstride=1, facecolors=plt.cm.Greys(-data), shade=False, alpha=0.1)


# %%
