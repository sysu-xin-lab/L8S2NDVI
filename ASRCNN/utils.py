import tensorflow.compat.v1 as tf
import csv
import os
import shutil
import scipy.io as io
import numpy as np
import pathlib as Path


def augmentation(image, randNumber):
    if randNumber == 0:
        result = image
    elif randNumber == 1:
        result = np.rot90(image, k=1)
    elif randNumber == 2:
        result = np.rot90(image, k=2)
    elif randNumber == 3:
        result = np.rot90(image, k=3)
    elif randNumber == 4:
        result = np.fliplr(image)  # np.fliplr for C*H*W
    elif randNumber == 5:
        result = np.flipud(image)  # np.flipud for C*H*W
    else:
        print('invaild randNumber!')
    return result


def readData(filename, rcstart, rcend, mode, data_scale):
    data = io.loadmat(filename)
    hr1 = data['hr1'].astype("float32") / data_scale
    hr2 = data['hr2'].astype("float32") / data_scale
    hr3 = data['hr3'].astype("float32") / data_scale
    lr1 = data['lr1'].astype("float32") / data_scale
    lr2 = data['lr2'].astype("float32") / data_scale
    lr3 = data['lr3'].astype("float32") / data_scale
    perm = data['perm'].astype("int").ravel() - 1
    hr1 = hr1[rcstart:rcend, rcstart:rcend, 0:4]
    hr2 = hr2[rcstart:rcend, rcstart:rcend, 0:4]
    hr3 = hr3[rcstart:rcend, rcstart:rcend, 0:4]
    lr1 = lr1[rcstart:rcend, rcstart:rcend, 0:4]
    lr2 = lr2[rcstart:rcend, rcstart:rcend, 0:4]
    lr3 = lr3[rcstart:rcend, rcstart:rcend, 0:4]

    hr1NDVI = computeNDVI(hr1)
    hr2NDVI = computeNDVI(hr2)
    hr3NDVI = computeNDVI(hr3)
    lr1NDVI = computeNDVI(lr1)
    lr2NDVI = computeNDVI(lr2)
    lr3NDVI = computeNDVI(lr3)

    NDVI = np.concatenate([lr1NDVI, lr2NDVI, lr3NDVI, hr1NDVI, hr3NDVI])
    minNDVI = np.min(NDVI)
    maxNDVI = np.max(NDVI)
    lr1NDVI = (lr1NDVI - minNDVI) / (maxNDVI - minNDVI)
    lr2NDVI = (lr2NDVI - minNDVI) / (maxNDVI - minNDVI)
    lr3NDVI = (lr3NDVI - minNDVI) / (maxNDVI - minNDVI)
    hr1NDVI = (hr1NDVI - minNDVI) / (maxNDVI - minNDVI)
    hr2NDVI = (hr2NDVI - minNDVI) / (maxNDVI - minNDVI)
    hr3NDVI = (hr3NDVI - minNDVI) / (maxNDVI - minNDVI)
    if mode == 'BI':
        trainData = np.concatenate([lr1, hr1, lr3], axis=-1)
        valData = np.concatenate([lr3, hr3, lr1], axis=-1)
        testData = np.concatenate([lr1, hr1, lr2], axis=-1)
    if mode == 'IB':

        trainData = np.concatenate([lr1NDVI, hr1NDVI, lr3NDVI], axis=-1)
        valData = np.concatenate([lr3NDVI, hr3NDVI, lr1NDVI], axis=-1)
        testData = np.concatenate([lr1NDVI, hr1NDVI, lr2NDVI], axis=-1)

    return trainData, testData, valData, hr1NDVI, hr2NDVI, hr3NDVI, minNDVI, maxNDVI, perm


def load_ckpt(sess, ckpt_dir, saver):

    # Require only one checkpoint in the directory.
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('Restoring from', ckpt_dir / ckpt_name)
        saver.restore(sess, str(ckpt_dir / ckpt_name))
        print('Successfully loaded checkpoint.')
        return True
    else:
        print('Failed to load checkpoint.')
        return False


def computeNDVI(image):
    ndvi = (image[:, :, 3] - image[:, :, 2]) / (image[:, :, 3] + image[:, :, 2])
    return ndvi[..., np.newaxis]  # H*W*C


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
    empty = True
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def computeLoss(phTarget, phPredction):

    # MSE = tf.reduce_sum((phTarget - phPredction)**2)
    RMSE = tf.reduce_mean((phTarget - phPredction)**2)**0.5
    SSIM = tf.reduce_mean(tf.image.ssim(phTarget, phPredction, max_val=1.0))
    return RMSE + (1 - SSIM)
