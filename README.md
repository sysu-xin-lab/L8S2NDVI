# L8S2NDVI
Fusion of Sentinel-Landsat NDVI using ASRCNN

The used data are publicly accessed via MS OneDrive: https://1drv.ms/u/s!AuRyYwtUQzjOhzQ8Xz3hiApcFNff?e=i6e5FW

Attached file namely "data.mat" includes the high resolution Sentinel-2(hr1,hr2,hr3) and low resolution landsat-8(lr1,lr2,lr3) images. The pre-processing of cloudmask and gapfilling has been conducted in MATLAB. Surface reflectences for all images ranges from 0~10000. The variables R and refCode can be used to  reconstruct the Geotiff file using the following matlab code:

geotiffwrite('Sentinel_t1.tif',hr1,R,'CoordRefSysCode',refCode);
geotiffwrite('Landsat_t2.tif',;lr2,R,'CoordRefSysCode',refCode);

Images in the "data.mat" have a size of 1600x1600. A subset was used for train and test, i.e, data[4:1596,4:1596,:]

Reproduce the results:
--------

Taking the BI mode for cropland site as example, copy the "data.mat" for cropland site to the folder "cropland_BI", and run 

    python asrcnn.py --area cropland --mode BI --test

Finally, the following result will be produced:

    [cropland-BI] test complete! Time used: 2.261s
    R2: 0.852461 RMSE: 0.081621 SSIM: 0.760000

Train a new model:
--------

delete "cropland_BI\history.csv" and run

    python asrcnn.py --area cropland --mode BI --train
