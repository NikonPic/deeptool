# %%
# hide
from nbdev.export import notebook2script

# Apply latest updates on the python files
notebook2script()
%matplotlib qt

# %%
from deeptool.train_loop import get_model, test_one_batch, main_loop
from deeptool.parameters import get_all_args, compat_args
from scipy.ndimage.interpolation import affine_transform


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
args.pic_size = 64

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
    mri = data['img']['axial'][0, :, :, :].numpy()
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
fig = plt.figure(figsize=(6, 8))
fig.set_facecolor('black')
ax = fig.add_subplot(211, projection='3d')
ax.set_facecolor('black')
ax.grid(False) 
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False


crop_size = mri.shape[0]
alp_max = 0.3
alp_min = 0.05
alp = np.linspace(alp_max, alp_min, crop_size)

for i in range(crop_size):
    z_loc = Z.copy()
    xx_loc = xx.copy()
    yy_loc = yy.copy()

    lim = 1

    for row, (z_row, dat_row) in enumerate(zip(z_loc, data)):
        for col, (z, dat) in enumerate(zip(z_row, dat_row)):
            z_loc[row, col] = np.NaN if dat < lim else z
            xx_loc[row, col] = np.NaN if dat < lim else xx[row, col]
            yy_loc[row, col] = np.NaN if dat < lim else yy[row, col]

    data = mri[i, :, :]

    ax.plot_surface(xx_loc, yy_loc, z_loc *0.1*i, rstride=1, cstride=1, facecolors=plt.cm.gray(data), shade=False, alpha=0.05)


x = mri
scale=0.7
alpha=1.0
bg_val=-1

images = [x[i, :, :] for i in range(crop_size)]

# Define size of new picture
stacked_height = 2 * args.pic_size
stacked_width = int(
    args.pic_size + (args.crop_size - 1) * args.pic_size * scale
)
stacked = np.full((stacked_height, stacked_width), bg_val)
T = np.array([[1, -1], [0, 1]])

# Go over each slide
for i in range(crop_size):
    # The first image will be right most and on the "bottom" of the stack.
    o = (crop_size - i - 1) * args.pic_size * scale
    out = affine_transform(
        images[i][:, :],
        T,
        offset=[o, -o],
        output_shape=stacked.shape,
        cval=bg_val,
    )
    stacked[out != bg_val] = out[out != bg_val]

ax2 = fig.add_subplot(212)
# plot the image series
ax2.imshow(stacked, alpha=alpha, interpolation="nearest", cmap="gray")


# %%
