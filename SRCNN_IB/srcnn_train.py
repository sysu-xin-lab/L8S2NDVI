import tensorflow as tf
from srcnnConfig import *
from utils import augmentation, readData, cropBound
import numpy as np
import os
import time
import utils
import random

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
num_epoch = train_num_epoch
batch_size = train_batch_size
model_dir = "%s_%s_mband" % (model_name, size_label)

# import_and_calibration
train1, train2, _, _, hr1_ndvi, _, hr3_ndvi, _, _ = readData(work_dir + 'data.mat', scale)

size_in_channels = train1.shape[-1]
sz = test_num_epoch * test_batch_size
perm = [i for i in range(sz * 2)]
batch_curr_pos = 0


def gen_train():
    global perm
    global batch_curr_pos

    result = np.zeros([batch_size, size_input, size_input, size_in_channels], dtype=np.float32)
    label = np.zeros([batch_size, size_label, size_label], dtype=np.float32)

    for i in range(batch_size):
        if batch_curr_pos == 0:
            random.shuffle(perm)
        idx = perm[batch_curr_pos]
        if idx < sz:
            temp = train1
            templ = hr3_ndvi
            r = (idx // test_num_epoch) * size_label + size_crop + rcstart
            c = (idx % test_batch_size) * size_label + size_crop + rcstart
        else:
            temp = train2
            templ = hr1_ndvi
            r = ((idx - sz) // test_num_epoch) * size_label + size_crop + rcstart
            c = ((idx - sz) % test_batch_size) * size_label + size_crop + rcstart
        randNumber = np.random.randint(0, 6)
        result[i, :, :, :] = augmentation(
            temp[r - size_crop:r + size_label + size_crop, c - size_crop:c + size_label +
                 size_crop, :], randNumber)
        label[i, :, :] = augmentation(templ[r:r + size_label, c:c + size_label], randNumber)
        batch_curr_pos = (batch_curr_pos + 1) % (sz * 2)
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


# Initialization.
tf.reset_default_graph()

# variables
train_images = tf.placeholder(tf.float32, [None, size_input, size_input, size_in_channels],
                              name='train_images')
train_labels = tf.placeholder(tf.float32, [None, size_label, size_label], name='train_labels')
global_steps = tf.Variable(0, name="global_step", trainable=False)

predctions = utils.cropBound(model(train_images))
# stas_graph
'''
graph = tf.get_default_graph()
flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
params = tf.profiler.profile(
    graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
'''
loss = tf.losses.mean_squared_error(labels=tf.squeeze(train_labels), predictions=predctions)
'''
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss,
                                                                         global_step=global_steps)
'''
optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(
    learning_rate, global_steps, decay_step, decay_ratio)).minimize(loss, global_step=global_steps)

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()
gpu_options = tf.GPUOptions(allow_growth=allow_growth)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    file_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, model_dir), sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    if load_ckpt(sess, ckpt_dir, saver):
        print('Successfully loaded checkpoint.')
    else:
        print('Failed to load checkpoint.')

    print('Training ...')
    start_time = time.time()
    # Training
    for epoch in range(num_epoch):
        epoch_images, epoch_labels = gen_train()
        _, epoch_loss, g_step = sess.run([optimizer, loss, global_steps],
                                         feed_dict={
                                             train_images: epoch_images,
                                             train_labels: epoch_labels
                                         })

        # Save the checkpoint every 500 steps.
        if g_step % 500 == 0:
            summary = sess.run(merged,
                               feed_dict={
                                   train_images: epoch_images,
                                   train_labels: epoch_labels
                               })
            file_writer.add_summary(summary, g_step)
            print(' epoch： ', g_step, 'loss:', epoch_loss, ' duration:', time.time() - start_time)

        if g_step % 1000 == 0:
            save_ckpt(sess, g_step, saver)

        if g_step > num_epoch:
            print(' training completed! ', 'loss:', epoch_loss, 'duration:',
                  time.time() - start_time)
            break
