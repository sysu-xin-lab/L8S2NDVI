# L8S2NDVI
Fusion of Sentinel-Landsat NDVI using SRCNN

Attached file namely "data.mat" shows the high resolution Sentinel-2(hr1,hr2,hr3) and low resolution landsat-8(lr1,lr2,lr3) images.

The pre-processing of cloudmask and gapfilling has been conducted in MATLAB. Surface reflectences for all images ranges from 0~10000.

Orginal images in the "data.mat" have a size of 1600x1600. A subset was used for train and test, i.e, data[4:1596,4:1596,:]

Train:

    python srcnn_train.py

Test:

    python srcnn_test.py

Evaluation using MATLAB:

    load('data.mat','refer_ndvi')

    load('pred-srcnn-NDVI.mat','pred')

    Metrics = computeMetric(refer_ndvi,pred); 
