# L8S2NDVI
This repository contains codes for spatiotemproal fusion of Sentinel-Landsat NDVI using ASRCNN

For more details, see: 
```
@ARTICLE{9125996,
  author={Ao, Zurui and Sun, Ying and Xin, Qinchuan},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Constructing 10-m NDVI Time Series From Landsat 8 and Sentinel 2 Images Using Convolutional Neural Networks}, 
  year={2021},
  volume={18},
  number={8},
  pages={1461-1465},
  doi={10.1109/LGRS.2020.3003322}}
```

Data:
--------

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
