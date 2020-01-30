# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\02_utils.ipynb (unless otherwise specified).

__all__ = ['Tracker']

# Cell

# For storing and Documentation
import datetime
import json
import datetime
import torch
import os
from tqdm import tqdm
from sklearn import metrics
from statistics import mean

# For Visualization
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from scipy.ndimage.interpolation import affine_transform

# Cell


class Tracker(object):
    """
    The Data tracker class serves as a uniform datatracker for all Modules
    """

    def __init__(self, args, log_view=True):
        """
        Setup the Folders for tracking
        """
        super(Tracker, self).__init__()
        # general defines:
        self.channels = len(args.perspectives)
        self.dim = args.dim
        self.crop_size = args.crop_size
        self.pic_size = args.pic_size
        self.model_type = args.model_type
        self.log_view = log_view
        # Affine transform matrix.
        self.T = np.array([[1, -1],
                           [0, 1]])
        # The results to track as empty dict:
        self.tr_dict = {}
        # create directory name
        date = datetime.datetime.now()
        self.dir_name = "../data/%s %d-%d-%d at %d-%d" % (
            args.model_type, date.year, date.month, date.day, date.hour, date.minute)

        # internal counting for visualization
        self.internal_count = 100
        self.classes = args.classes
        self.best_score = {}
        self.best_score["all"] = 1e8
        for cl in self.classes:
            self.best_score[cl] = 1e8
        # store the path of the best model
        self.model_path = self.dir_name + "/_model"

        # repair for incorrect view
        self.view_re_x = [1, len(args.perspectives), -1,
                          args.pic_size, args.pic_size]

        # Create new directory
        try:
            os.mkdir(self.dir_name)
        except:
            print("\nFoler: " + self.dir_name +
                  " exists already.\nFiles will be overwritten.\n")
            open(self.dir_name + "\_Log.txt", 'w').close()

        # Save set of hyperparameters
        with open(self.dir_name + '/_params.json', 'w') as f:
            json.dump(args._get_kwargs(), f, ensure_ascii=False, indent=4)

    def view_3d(self, x, scale=0.7, alpha=1.0, bg_val=-1):
        """
        Visualize the 3d pictures as expanded slideshow
        args:
            x: 3d matrix (img_h, img_w, img_z)
            sclae: define the distance between slides
            alpha: visibility
            bg_val: background value
        """
        # List of images instead 3d matrix
        x = x.numpy()
        images = [x[i, :, :] for i in range(self.crop_size)]

        # Define size of new picture
        stacked_height = 2*self.pic_size
        stacked_width = int(
            self.pic_size + (self.crop_size-1)*self.pic_size * scale)
        stacked = np.full((stacked_height, stacked_width), bg_val)

        # Go over each slide
        for i in range(self.crop_size):
            # The first image will be right most and on the "bottom" of the stack.
            o = (self.crop_size-i-1) * self.pic_size * scale
            out = affine_transform(images[i][0, :, :], self.T, offset=[o, -o],
                                   output_shape=stacked.shape, cval=bg_val)
            stacked[out != bg_val] = out[out != bg_val]

        # plot the image series
        plt.imshow(stacked, alpha=alpha, interpolation='nearest', cmap='gray')

    def show(self, img):
        """subfunction for visualization"""
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    def track_progress(self, model, data, iteration):
        """
        Visualize the Progress by applying the validation loader and visualizing the results
        """
        # Get the original image
        x = data["img"].detach().cpu()

        # Get the current results:
        x_re, tr_data = model(data, update=False)
        x_re = x_re.detach().cpu()

        # Go over all Losses and Plot
        plt.figure(figsize=(12, 12))
        for k in tr_data.keys():
            # Append / Create new
            if k in self.tr_dict.keys():
                self.tr_dict[k].append(tr_data[k])
            else:
                self.tr_dict[k] = [tr_data[k]]

            # Plot the current progress in "_Loss.png"
            plt.plot(self.tr_dict[k], label=k)

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        if self.log_view:
            plt.yscale('log')
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.savefig(self.dir_name + "/_Losses.pdf")
        plt.close()

        # write to "_Log.txt" file
        with open(self.dir_name + "/_Log.txt", "a") as file:
            file.write("\n\nIteration: %d" % (iteration))
            for k in tr_data.keys():
                file.write("\n%s: %.4f" % (k, tr_data[k]))

        # Save the losses as npy
        np.save(self.dir_name + "/_Losses.npy", self.tr_dict)

        # save the current model:
        torch.save(model.state_dict(), self.dir_name + "/_model")

        # Plot Results:
        if self.model_type not in ("diagnosis") and self.internal_count > 2:
            self.internal_count = 0

            for channel in range(self.channels):
                # init view list
                if self.dim == 3:

                    if len(x.shape) < 5:
                        x = x.permute(1, 0, 2, 3)
                        x = x.reshape(self.view_re_x)

                    if len(x_re.shape) < 5:
                        x_re = x_re.permute(1, 0, 2, 3)
                        x_re = x_re.reshape(self.view_re_x)

                    real_pic = x[0, channel, :, :, :].view(
                        (self.crop_size, -1, self.pic_size, self.pic_size))

                    fake_pic = x_re[0, channel, :, :, :].view(
                        (self.crop_size, -1, self.pic_size, self.pic_size))

                    # add the 3d visualization
                    plt.figure(figsize=(24, 12))
                    plt.subplot(2, 1, 1)
                    self.view_3d(real_pic)
                    plt.subplot(2, 1, 2)
                    self.view_3d(fake_pic)
                    plt.savefig(self.dir_name + "/rec_iteration_%d_class_%d_3d_view.png" %
                                (iteration, channel))
                    plt.close()

                else:
                    real_pic = x[0, channel, :, :].view(
                        (self.crop_size, -1, self.pic_size, self.pic_size))
                    fake_pic = x_re[0, channel, :, :].view(
                        (self.crop_size, -1, self.pic_size, self.pic_size))

                # plot in figure
                plt.figure(channel, figsize=(24, 24))
                if iteration == 0:
                    self.show(vutils.make_grid(
                        real_pic, nrow=4, padding=2, normalize=True))
                    plt.savefig(self.dir_name + "/class_%d__original.jpg" %
                                (channel))
                else:
                    self.show(vutils.make_grid(
                        fake_pic, nrow=4, padding=2, normalize=True))
                    plt.savefig(self.dir_name + "/rec_iteration_%d_class_%d.jpg" %
                                (iteration, channel))
                plt.close()

        # increase the counter
        self.internal_count += 1

    def update_network(self, cl, model, score):
        """
        update the network if the score is lower
        """
        if self.best_score[cl] > score:
            print("\n\n  ---Best Score %s %.4f---- \n" % (cl, 1-score))
            # delete old model if it exists
            if self.best_score[cl] < 1:
                os.remove(self.model_path + "_%s_%.4f" %
                          (cl, 1-self.best_score[cl]))
            # save new model
            torch.save(model.state_dict(), self.model_path +
                       "_%s_%.4f" % (cl, 1-score))
            # reset score
            self.best_score[cl] = score

    def get_accuracy(self, model, valid_loader, iteration):
        """
        Go trough the whole validation dataset and determine accuracy
        """
        # predefine
        count = 0
        loss = 0
        preds = {}
        labels = {}
        for cl in self.classes:
            preds[cl] = []
            labels[cl] = []

        # calculate accuracy
        for data in valid_loader:
            pred, label, tr_data = model.forward(data, update=False)
            # get the current output
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
            # append predictions
            for i, cl in enumerate(self.classes):
                preds[cl].append(pred[i])
                labels[cl].append(label[i])

            loss += tr_data["loss"]
            count += 1

        # get the accuracies
        acc = {}
        for cl in self.classes:
            fpr, tpr, _ = metrics.roc_curve(labels[cl], preds[cl])
            acc[cl] = 1 - metrics.auc(fpr, tpr)

        acc["all"] = mean([acc[cl] for cl in self.classes])

        loss /= count

        if "err_all" in self.tr_dict.keys():
            for cl in acc.keys():
                self.tr_dict["err_%s" % (cl)].append(acc[cl])
        else:
            for cl in acc.keys():
                self.tr_dict["err_%s" % (cl)] = [acc[cl]]

        # plot result
        plt.figure(figsize=(12, 12))
        for k in self.tr_dict.keys():
            # Plot the current progress in "_Loss.png"
            plt.plot(self.tr_dict[k], label=k)

        # format plot
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.savefig(self.dir_name + "/_Losses.pdf")
        plt.close()

        # write to "_Log.txt" file
        with open(self.dir_name + "/_Log.txt", "a") as file:
            file.write("\n\nIteration: %d" % (iteration))
            for k in self.tr_dict.keys():
                file.write("\n%s: %.4f" % (k, self.tr_dict[k][-1]))

        # Save the losses as npy
        np.save(self.dir_name + "/_Losses.npy", self.tr_dict)

        # save the best performing networks
        for cl in acc.keys():
            self.update_network(cl, model, acc[cl])