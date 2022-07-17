import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt
from time import strptime, mktime
import stippy
import socket
import gdal
from skimage.transform import resize
from skimage.color import rgb2gray
import datetime

def convertDateToEpoch(inputDate):
    dateF = strptime(inputDate + ' 00:00', '%Y-%m-%d %H:%M')
    epochTime = mktime(dateF)
    return int(epochTime)


def convertEpochToDate(inputEpoch):
    return datetime.datetime.fromtimestamp(inputEpoch).strftime('%Y-%m-%d')


def getTime(startT, endT):
    if startT is not None:
        startT = convertDateToEpoch(startT)
    if endT is not None:
        endT = convertDateToEpoch(endT)
    return startT, endT

def accessFiles(paths):
            loadedImages = []
            loaded_image = gdal.Open(paths)
            image = loaded_image.ReadAsArray() #(11, h, w)
            image = np.moveaxis(image, 0, -1)  #(h, w, 11)

            factor = 2 / 1e4


            image = image * factor

            image = resize(image, (128,128), preserve_range=True) #( 128, 128)
            # image = self.scale_images_11(image)
            loadedImages.append(image[:,:,3])
            return loadedImages,image[:,:,:3]

def getSentinelTileAtGivenTime( maxSentCloud=0.1):
    startT, endT = getTime('2020-05-01', '2021-08-01')

    listImages = stippy.list_node_images(socket.gethostbyname(socket.gethostname()) + ':15606', album='iowa-2015-2020', geocode=None, recurse=False,
                                         platform='Sentinel-2', max_cloud_coverage=maxSentCloud, min_pixel_coverage=0.9,
                                         start_timestamp=startT, end_timestamp=endT)

    g =[]
    try:
        (node, b_image) = listImages.__next__()
        while True:
            for p in b_image.files:
                if p.path.endswith('-0.tif'):
                    pathf = p.path
                    if b_image.geocode in g:
                        continue
                    else:
                        yield pathf
                        g.append(b_image.geocode)
            (_, b_image) = listImages.__next__()

    except StopIteration:
        return None



def load_edge(img, index):
        sigma = self.sigma

        mask = None

        # # canny
        # if self.edge == 1:
        #     # no edge
        #     if sigma == -1:
        #         return np.zeros(img.shape).astype(np.float)
        #
        #     # random sigma
        #     if sigma == 0:
        #         sigma = random.randint(1, 4)
        #
        #     return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        # else:
        imgh, imgw = img.shape[0:2]
        edge = imread(self.edge_data[index])
        edge = self.resize(edge, imgh, imgw)

        # non-max suppression
        # if self.nms == 1:
        edge = edge * canny(img, sigma=sigma, mask=mask)

        return edge

if __name__ == '__main__':
    it = getSentinelTileAtGivenTime(0.05)
    epoch = 0
    for p in it:
        if epoch > 50:
            break

        _, image1 = accessFiles(p)
        image = rgb2gray(image1)
        epoch +=1

        # edges1 = canny(image)
        # edges2 = canny(image, sigma=1.1)
        # edges3 = canny(image, sigma=1.2)
        edges4 = canny(image, sigma=0)
        # edges5 = canny(image, sigma=1.4)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(image1)
        ax[0].set_title('Input image', fontsize=10)

        ax[1].imshow(edges4, cmap='gray')
        ax[1].set_title(r'$\sigma=0.1$', fontsize=10)

        # ax[2].imshow(edges2, cmap='gray')
        # ax[2].set_title(r'$\sigma=1.1$', fontsize=8)
        #
        # ax[3].imshow(edges3, cmap='gray')
        # ax[3].set_title(r'$\sigma=1.2$', fontsize=8)
        #
        # ax[4].imshow(edges4, cmap='gray')
        # ax[4].set_title(r'$\sigma=1.3$', fontsize=8)
        #
        # ax[5].imshow(edges5, cmap='gray')
        # ax[5].set_title(r'$\sigma=1.4$', fontsize=8)

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        fig.savefig('./sen/' + "%s.png" % (epoch))
        plt.close()




