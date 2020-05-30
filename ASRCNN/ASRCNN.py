import tensorflow.compat.v1 as tf
from pathlib import Path
import math
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as io
import csv
import os
import time
import shutil
import utils
import argparse
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity
tf.disable_v2_behavior()
'''
python asrcnn.py --area forest --mode IB --train --test
python asrcnn.py --area cropland --mode IB --gpuID 1 --train --test
'''

parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--train', action='store_true', help='enables train')
parser.add_argument('--test', action='store_true', help='enables test')
parser.add_argument('--gpuID', type=str, default='0', help='GPU ID')
parser.add_argument('--mode', type=str, choices=['IB', 'BI'], help='IB or BI')
parser.add_argument('--area', type=str, choices=['forest', 'cropland'], help='study area')
opt = parser.parse_args()

if opt.mode == 'BI':
    NUM_BANDS_IN = 12  # 3 iamges, each containing 4 bands(NIR-RGB)
if opt.mode == 'IB':
    NUM_BANDS_IN = 3  # 3 iamges, each containing 1 band(NDVI)

# modify start
data_scale = 10000
lr = 1e-4
rcstart = 4
rcend = 1596
image_size = rcend - rcstart
batch_size = 128
patch_size = 32
patch_stride = 20
crop_size = (patch_size - patch_stride) // 2

num_epochs = 1000
decay_step = 10000
decay_ratio = 0.7
max_to_keep = None
allow_growth = True

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuID
model_name = 'ASRCNN'
data_dir = Path('.') / f'{opt.area}_{opt.mode}'
data_file = data_dir / 'data.mat'
result_file = f'{model_name}-{opt.area}-{opt.mode}.mat'

ckpt_dir = data_dir / 'checkpoint'
ckpt_dir.mkdir(exist_ok=True)
best_ckpt_file = ckpt_dir / f'{model_name}-best.ckpt'
history_file = data_dir / 'history.csv'


def conv2d(net, filters, kernel_size=3, strides=1, activation=None, padding='same', name='conv'):
    return tf.layers.conv2d(net,
                            filters,
                            kernel_size,
                            strides=strides,
                            activation=activation,
                            padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001),
                            name=name)


def sam(net, filters, name):
    skip = net
    net = conv2d(net, filters, activation='relu', name=name + 'conv3x3')
    mask = conv2d(net, filters, activation='sigmoid', name=name + 'mask')
    return skip + net * mask


# model
def ASRCNN(net):
    net = conv2d(net, 64, 9, name='conv1')
    net = sam(net, 64, name='sam1')
    net = tf.nn.relu(net)
    net = conv2d(net, 32, 3, name='conv2')
    net = sam(net, 32, name='sam2')
    net = tf.nn.relu(net)
    net = conv2d(net, 1, 5, name='conv3')
    return net


tf.reset_default_graph()
phTrainInput = tf.placeholder(tf.float32, [None, patch_size, patch_size, NUM_BANDS_IN],
                              name='train_images')
phTrainTarget = tf.placeholder(tf.float32, [None, patch_size, patch_size, 1], name='train_labels')

global_steps = tf.Variable(0, name="global_step", trainable=False)
phPredction = ASRCNN(phTrainInput)
loss = utils.computeLoss(phTrainTarget, phPredction)
curr_lr_op = tf.train.exponential_decay(lr, global_steps, decay_step, decay_ratio, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate=curr_lr_op).minimize(loss, global_step=global_steps)
gpu_options = tf.GPUOptions(allow_growth=allow_growth)

# data
trainData1, testData, trainData2, trainTarget2, testTarget, trainTarget1, minNDVI, maxNDVI, perm = utils.readData(
    data_file, rcstart, rcend, opt.mode, data_scale)

trainData = [trainData1, trainData2]
trainTarget = [trainTarget1, trainTarget2]
num_patches_x = (image_size - patch_size + patch_stride) // patch_stride
num_patches_y = (image_size - patch_size + patch_stride) // patch_stride
num_patches = num_patches_x * num_patches_y

print(f'Extracting 80% for training,and 20% for validation ...')
pos = np.int(np.ceil(num_patches * 2 * 0.2 / batch_size) * batch_size)
valPerm = perm[:pos]
trainPerm = perm[pos:]
start_time = time.time()


def train_one_epoch(sess, n_epoch, saver):
    epoch_loss = utils.AverageMeter()
    np.random.shuffle(trainPerm)
    for i in range(trainPerm.shape[0] // batch_size):
        batchTrain = np.zeros([batch_size, patch_size, patch_size, NUM_BANDS_IN], dtype=np.float32)
        batchTarget = np.zeros([batch_size, patch_size, patch_size, 1], dtype=np.float32)
        for j in range(batch_size):
            id_n = trainPerm[i * batch_size + j] // num_patches
            residual = trainPerm[i * batch_size + j] % num_patches
            id_x = patch_stride * (residual % num_patches_x)
            id_y = patch_stride * (residual // num_patches_x)
            image = trainData[id_n][id_x:id_x + patch_size, id_y:id_y + patch_size, :]
            augmentationType = np.random.randint(0, 6)
            batchTrain[j] = utils.augmentation(image, augmentationType)
            image = trainTarget[id_n][id_x:id_x + patch_size, id_y:id_y + patch_size, :]
            batchTarget[j] = utils.augmentation(image, augmentationType)
        _, batchLoss, g_step, curr_lr = sess.run([train_op, loss, global_steps, curr_lr_op],
                                                 feed_dict={
                                                     phTrainInput: batchTrain,
                                                     phTrainTarget: batchTarget
                                                 })
        epoch_loss.update(batchLoss)
    total_loss = epoch_loss.avg
    saver.save(sess, str(ckpt_dir / f'{model_name}-{n_epoch}.ckpt'))
    return total_loss, curr_lr


def val_one_epoch(sess, n_epoch, saver, best_acc):
    epoch_loss = utils.AverageMeter()
    for i in range(valPerm.shape[0] // batch_size):
        batchTrain = np.zeros([batch_size, patch_size, patch_size, NUM_BANDS_IN], dtype=np.float32)
        batchTarget = np.zeros([batch_size, patch_size, patch_size, 1], dtype=np.float32)
        for j in range(batch_size):
            id_n = valPerm[i * batch_size + j] // num_patches
            residual = valPerm[i * batch_size + j] % num_patches
            id_x = patch_stride * (residual % num_patches_x)
            id_y = patch_stride * (residual // num_patches_x)
            image = trainData[id_n][id_x:id_x + patch_size, id_y:id_y + patch_size, :]
            augmentationType = np.random.randint(0, 6)
            batchTrain[j] = utils.augmentation(image, augmentationType)
            image = trainTarget[id_n][id_x:id_x + patch_size, id_y:id_y + patch_size, :]
            batchTarget[j] = utils.augmentation(image, augmentationType)
        batchLoss = sess.run(loss, feed_dict={phTrainInput: batchTrain, phTrainTarget: batchTarget})
        epoch_loss.update(batchLoss)
    total_loss = epoch_loss.avg
    if total_loss <= best_acc:
        best_acc = total_loss
        saver.save(sess, str(best_ckpt_file))
    return total_loss, best_acc


def train():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    utils.load_ckpt(sess, ckpt_dir, saver)
    best_loss = 1e6
    start_epoch = 0
    if history_file.exists():
        df = pd.read_csv(history_file)
        best_loss = df['best_loss'].min()
        start_epoch = int(df.iloc[-1]['epoch']) + 1

    print('Training ...')
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_lr = train_one_epoch(sess, epoch, saver)
        val_loss, best_loss = val_one_epoch(sess, epoch, saver, best_loss)
        csv_header = ['epoch', 'lr', 'train_loss', 'val_loss', 'best_loss']
        csv_values = [epoch, train_lr, train_loss, val_loss, best_loss]
        utils.log_csv(history_file, csv_values, header=csv_header if epoch == 0 else None)
        print(
            f'[{opt.area}-{opt.mode}] Epoch {epoch} loss:{train_loss:.6f}, val loss:{val_loss:.6f},duration:{time.time() - start_time:.3f}s'
        )

    print('Training completed...')


def test():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        file_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver.restore(sess, str(best_ckpt_file))
        #
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        # final result
        result = np.zeros([num_patches_x * patch_stride, num_patches_y * patch_stride],
                          dtype=np.float32)
        # batch data
        batchTest = np.zeros([num_patches_y, patch_size, patch_size, NUM_BANDS_IN],
                             dtype=np.float32)
        print('Testing ...')
        start_time = time.time()
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                id_x = patch_stride * j
                id_y = patch_stride * i
                batchTest[j] = testData[id_x:id_x + patch_size, id_y:id_y + patch_size, :]

            batchPred = sess.run(phPredction, feed_dict={phTrainInput: batchTest})
            batchPred = batchPred[:, crop_size:-crop_size, crop_size:-crop_size, 0]

            for j in range(num_patches_y):
                id_x = patch_stride * j
                id_y = patch_stride * i
                result[id_x:id_x + patch_stride, id_y:id_y + patch_stride] = batchPred[j]

        end_time = time.time()
        refer = testTarget[crop_size:-crop_size, crop_size:-crop_size, 0]
        refer = refer * (maxNDVI - minNDVI) + minNDVI
        result = result * (maxNDVI - minNDVI) + minNDVI
        total_RMSE = ((refer - result)**2).mean()**.5
        r2 = r2_score(refer.ravel(), result.ravel())
        SSIM = structural_similarity(refer, result, data_range=1.0)
        io.savemat(result_file, {'refer': refer, 'pred': result})
        print(f'[{opt.area}-{opt.mode}] test complete! Time used: {end_time-start_time:.3f}s')
        print(f'R2: {r2:.6f}', f'RMSE: {total_RMSE:.6f}', f'SSIM: {SSIM:.6f}')


if __name__ == '__main__':
    if opt.train:
        train()
    if opt.test:
        test()
