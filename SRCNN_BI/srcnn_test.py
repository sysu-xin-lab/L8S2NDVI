import tensorflow as tf
from srcnnConfig import *
from utils import augmentation, readData, cropBound
import numpy as np
import os
import time
import utils
import random
import scipy.io as io

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
num_epoch = test_num_epoch
batch_size = test_batch_size
model_dir = "%s_%s_mband" % (model_name, size_label)

# import_and_calibration
_, _, test, _, _, hr2_ndvi, _, minNDVI, maxNDVI = readData(work_dir + 'data.mat', scale)

size_in_channels = test.shape[-1]


def gen_test():
    row = epoch
    result = np.zeros([batch_size, size_input, size_input, size_in_channels], dtype=np.float32)
    label = np.zeros([batch_size, size_label, size_label, size_out_channels], dtype=np.float32)
    r = row * size_label + size_crop + rcstart
    for idx in range(0, batch_size):
        c = idx * size_label + size_crop + rcstart
        result[idx, :, :, :] = test[r - size_crop:r + size_label + size_crop, c - size_crop:c +
                                    size_label + size_crop, :]
        label[idx, :, :, 0] = hr2_ndvi[r:r + size_label, c:c + size_label]
    return result, label


def load_ckpt(sess, checkpoint_dir, saver):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    print('checkpoint_dir is', checkpoint_dir)

    # Require only one checkpoint in the directory.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('Restoring from', os.path.join(checkpoint_dir, ckpt_name))
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


# Save the current checkpoint
def save_ckpt(sess, step, saver):
    checkpoint_dir = os.path.join(ckpt_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


tf.reset_default_graph()
test_images = tf.placeholder(tf.float32, [None, size_input, size_input, size_in_channels],
                             name='test_images')
test_labels = tf.placeholder(tf.float32, [None, size_label, size_label, size_out_channels],
                             name='test_labels')
predctions = utils.cropBound(model(test_images))
pred = np.zeros([size_label * batch_size, size_label * batch_size])
l2_loss = tf.losses.mean_squared_error(labels=tf.squeeze(test_labels), predictions=predctions)

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    print('Test ...')
    start_time = time.time()
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    if load_ckpt(sess, ckpt_dir, saver):
        print('Successfully loaded checkpoint.')
    else:
        print('Failed to load checkpoint.')
    # test
    for epoch in range(num_epoch):
        epoch_images, epoch_labels = gen_test()
        epoch_pred, epoch_loss = sess.run([predctions, l2_loss],
                                          feed_dict={
                                              test_images: epoch_images,
                                              test_labels: epoch_labels
                                          })
        r = epoch * size_label
        for idx in range(0, batch_size):
            c = idx * size_label
            pred[r:r + size_label, c:c + size_label] = epoch_pred[idx, :, :]

        if epoch % 50 == 0:
            print('Epoch:', epoch, 'loss:', epoch_loss, 'duration:', time.time() - start_time)

    refer = io.loadmat(work_dir + 'data.mat')
    refer = refer['refer_ndvi']
    pred = np.clip(pred, 0, 1) * (maxNDVI - minNDVI) + minNDVI
    total_loss = ((refer - pred)**2).mean()
    io.savemat('pred-{}-NDVI.mat'.format(model_name), {'pred': pred})
    print('test complete! total_loss:', total_loss)
