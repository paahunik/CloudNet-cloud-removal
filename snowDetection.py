import os
import s2cloudless
import numpy as np
import sys
import gdal
from matplotlib import pyplot as plt
from skimage.transform import resize

def getNDSImap(green, swir):
        return (green - swir) / (green + swir)

def getSnowCoverge(self, ndsiMap, redB):
    snowPixels = 0
    clearPixel = 0
    for i in range(ndsiMap.shape[0]):
        for j in range(ndsiMap.shape[1]):
            if ndsiMap[i][j] >= 0.400 and redB[i][j] >= 0.200: # Pass 1
                snowPixels += 1
            else:
                clearPixel += 1
    snowFrac = snowPixels / (clearPixel + snowPixels)

    snowPixels = 0
    clearPixel = 0
    if snowFrac >= 0.001:
        for i in range(ndsiMap.shape[0]):
            for j in range(ndsiMap.shape[1]):
                if ndsiMap[i][j] >= 0.150 and redB[i][j] >= 0.040:  # Pass 2
                    snowPixels += 1
                else:
                    clearPixel += 1
        snowFrac = snowPixels / (clearPixel + snowPixels)
    return snowFrac


def getNDVIMap(redb, nirb):
    return (nirb - redb) / (nirb + redb)

if __name__ == '__main__':

    # f1 = 'L1C_T14TQP_A019343_20201118T172617-'
    # f1 = 'L1C_T14TQP_A016154_20200409T172046-'
    # f1 = 'L1C_T14TQP_A015625_20200303T172459-'
    # f1 = 'L1C_T14TQP_A026178_20200626T172628-'
    f1 = 'L1C_T14TQP_A027036_20200825T173103-'
    p = './sentsamp/' + f1
    factor = 0.0002
    im = gdal.Open(p + '0.tif')
    rgbI = im.ReadAsArray().transpose(2, 1, 0)
    blueB = rgbI[:, :, 2] * factor
    greenB = rgbI[:, :, 1] * factor
    redB = rgbI[:,:, 0] * factor
    nirB = rgbI[:, :, 3] * factor

    rgbstacked = np.dstack((redB, greenB ,blueB))

    im = gdal.Open(p + '1.tif')
    swirI = im.ReadAsArray().transpose(2, 1, 0)
    swirB = swirI[:,:,-2] * factor
    swirB = resize(swirB, (rgbstacked.shape[0],rgbstacked.shape[1]), preserve_range=True)

    im = gdal.Open(p + '3.tif')
    rgbTCI = im.ReadAsArray().transpose(2, 1, 0)
    redTCIB = rgbTCI[:, :, 0]
    greenTCIB = rgbTCI[:, :, 1]
    blueTCIB = rgbTCI[:, :, 2]
    rgbstackedTCI = np.dstack((redTCIB, greenTCIB, blueTCIB))

    fig, axarr = plt.subplots(2, 2, figsize=(15, 12))
    np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

    axarr[0, 0].imshow(rgbstacked)
    axarr[0, 0].set_title('DN RGB SN:' + str(getSnowCoverge(getNDSImap(greenB, swirB), redB)), fontdict={'fontsize': 15})

    img = axarr[0, 1].imshow(getNDSImap(greenB, swirB), cmap=plt.cm.summer)
    fig.colorbar(img, ax=axarr[0,1])
    axarr[0, 1].set_title('NDSI MAP RGB >0.400 | >0.150', fontdict={'fontsize': 15})

    img2 = axarr[1, 0].imshow(getNDVIMap(redTCIB/255, nirB), cmap=plt.cm.summer)
    fig.colorbar(img2, ax=axarr[1,0])
    axarr[1, 0].set_title('NDVI MAP', fontdict={'fontsize': 15})


    img3 = axarr[1, 1].imshow(redB, cmap='hot')
    fig.colorbar(img3, ax=axarr[1, 1])
    axarr[1, 1].set_title('Red Band', fontdict={'fontsize': 15})


    fig.savefig('./sentsamp/5.png')
    plt.close()