from srcnn import srcnn

work_dir = './'
cuda_devices = '0'
learning_rate = 1e-4
# momentum = 0.9
train_batch_size = 128
train_num_epoch = 300000
decay_step = 50000
decay_ratio = 0.8
max_to_keep = None
allow_growth = True
scale = 10000

rcstart = 4
rcend = 1596
size_input = 32
size_crop = 6
mwWidth = 16
mwHeight = 16
size_label = size_input - 2 * size_crop
size_out_channels = 1

test_batch_size = (rcend - rcstart - 2 * size_crop) // size_label
test_num_epoch = test_batch_size

ckpt_dir = '{}checkpoint/'.format(work_dir)
model_name = 'srcnn'


def model(net, numOutChannels=size_out_channels):
    return srcnn(net, numOutChannels)
