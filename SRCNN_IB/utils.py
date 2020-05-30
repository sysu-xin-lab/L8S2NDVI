import tensorflow as tf
import scipy.io as io
import numpy as np
from srcnnConfig import size_crop


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
        result = np.fliplr(image)
    elif randNumber == 5:
        result = np.flipud(image)
    else:
        print('invaild randNumber!')
    return result


def cropBound(images, boundSize=size_crop):
    return tf.squeeze(images[:, boundSize:-boundSize, boundSize:-boundSize, :])


def readData(filename, scale):
    data = io.loadmat(filename)
    hr1 = data['hr1'].astype("float32") / scale
    hr2 = data['hr2'].astype("float32") / scale
    hr3 = data['hr3'].astype("float32") / scale
    lr1 = data['lr1'].astype("float32") / scale
    lr2 = data['lr2'].astype("float32") / scale
    lr3 = data['lr3'].astype("float32") / scale

    # hr index
    hr1_ndvi = (hr1[:, :, 3] - hr1[:, :, 2]) / (hr1[:, :, 3] + hr1[:, :, 2])
    hr2_ndvi = (hr2[:, :, 3] - hr2[:, :, 2]) / (hr2[:, :, 3] + hr2[:, :, 2])
    hr3_ndvi = (hr3[:, :, 3] - hr3[:, :, 2]) / (hr3[:, :, 3] + hr3[:, :, 2])
    # lr index
    lr1_ndvi = (lr1[:, :, 3] - lr1[:, :, 2]) / (lr1[:, :, 3] + lr1[:, :, 2])
    lr2_ndvi = (lr2[:, :, 3] - lr2[:, :, 2]) / (lr2[:, :, 3] + lr2[:, :, 2])
    lr3_ndvi = (lr3[:, :, 3] - lr3[:, :, 2]) / (lr3[:, :, 3] + lr3[:, :, 2])

    # lr index calibration
    minNDVI = np.min(np.dstack([lr1_ndvi, lr2_ndvi, lr3_ndvi]))
    maxNDVI = np.max(np.dstack([lr1_ndvi, lr2_ndvi, lr3_ndvi]))
    lr1_ndvi = (lr1_ndvi - minNDVI) / (maxNDVI - minNDVI)
    lr2_ndvi = (lr2_ndvi - minNDVI) / (maxNDVI - minNDVI)
    lr3_ndvi = (lr3_ndvi - minNDVI) / (maxNDVI - minNDVI)

    # hr index calibration, minNDVI maxNDVI are rewrited
    minNDVI = np.min(np.dstack([hr1_ndvi, hr3_ndvi]))
    maxNDVI = np.max(np.dstack([hr1_ndvi, hr3_ndvi]))
    hr1_ndvi = (hr1_ndvi - minNDVI) / (maxNDVI - minNDVI)
    hr2_ndvi = (hr2_ndvi - minNDVI) / (maxNDVI - minNDVI)
    hr3_ndvi = (hr3_ndvi - minNDVI) / (maxNDVI - minNDVI)

    train1 = np.dstack([hr1_ndvi, lr1_ndvi, lr3_ndvi])
    train2 = np.dstack([hr3_ndvi, lr3_ndvi, lr1_ndvi])
    test1 = np.dstack([hr1_ndvi, lr1_ndvi, hr2_ndvi])
    test2 = np.dstack([hr3_ndvi, lr3_ndvi, hr2_ndvi])
    return train1, train2, test1, test2, hr1_ndvi, hr2_ndvi, hr3_ndvi, minNDVI, maxNDVI
