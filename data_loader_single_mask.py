import os
import sys
from skimage.color import rgb2gray
import random
import gzip
import bz2
import gdal
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir + '/../')
import stippy
from time import strptime, mktime
import datetime
import socket
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
np.set_printoptions(threshold=sys.maxsize)
from skimage.transform import resize
import copy
from matplotlib import pyplot as plt
from skimage.feature import canny
import pickle
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
random.seed(42)

# TODO
# 1. Generate equal random masks for each image. 20-80% cloud.
# Experiment 1: Per cloud window accuracy -> ?
# Comparison between SpaNET, Resnet, CloudNET
# With clouds as inputs vs black cloud mask.
# Per Landcover type cloud coverage?

# cc: 20-30%
# cc: 30-40%
# cc: 40-50%
# cc: 50-60%
# cc: 60-70%
# cc: 70-80%

class DatasetHandling():

    def __init__(self, w=128, h=128, no_of_timesteps=1,
                 pix_cover=1.0,cloud_cov=0.3, album='iowa-2015-2020-spa', batch_size=1, istrain=True,
                 all_black_clouds = True, folderI = 10, saveInputMetaData = False):

        self.targetH = h
        self.targetW = w
        self.pix_cover = pix_cover
        self.cloud_cov = cloud_cov
        self.no_of_timesteps = no_of_timesteps
        self.album = album
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'

        self.ccThreshTest = 0.9
        self.ccThreshTrain = 0.2
        self.batch_size = batch_size
        self.lenarr=[]
        self.istrain = istrain
        self.all_black_clouds = all_black_clouds
        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(folderI) + "/"
        if saveInputMetaData:
            self.metaSentinel = self.getSentinelTileAtGivenTime()
            self.metaLandsat = self.getLansdsatTileAtGivenTime()
            if not os.path.isdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"):
                os.mkdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/")
            if not os.path.isdir(self.dirName):
                os.mkdir(self.dirName)
            self.train_and_test_landsat_paths()
            # self.save_input_output_files()
            self.save_input_output_files_validation()


    def convertDateToEpoch(self, inputDate):
        dateF = strptime(inputDate + ' 00:00', '%Y-%m-%d %H:%M')
        epochTime = mktime(dateF)
        return int(epochTime)

    def convertEpochToDate(self, inputEpoch):
        return datetime.datetime.fromtimestamp(inputEpoch).strftime('%Y-%m-%d')

    def getTime(self, startT, endT):
        if startT is not None:
            startT = self.convertDateToEpoch(startT)
        if endT is not None:
            endT = self.convertDateToEpoch(endT)
        return startT, endT

    def normalize11(self, imgs):
        """ Returns normalized images between (-1 to 1) pixel value"""
        return imgs / 127.5 - 1

    def denormalize11(self, imgs):
        return (imgs + 1.) * 127.5

    def normalize01(self, imgs):
            """ Returns normalized images between (0 to 1) pixel value"""
            return imgs / 255

    def denormalize01(self, imgs):
        return imgs * 255

    def scale_landsat_images(self, imgs):
        img_tmp = exposure.rescale_intensity(imgs,
                                             out_range=(1, 65535)).astype(np.int)
        img_tmp = np.sqrt(img_tmp).astype(np.int)
        return img_tmp

    def getNDSImap(self, green, swir):
        return (green - swir) / (green + swir)

    def getSnowCoverge(self, ndsiMap, redB):
        snowPixels = 0
        clearPixel = 0
        for i in range(ndsiMap.shape[0]):
            for j in range(ndsiMap.shape[1]):
                if ndsiMap[i][j] >= 0.400 and redB[i][j] >= 0.200:  # Pass 1
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

    def getCompressRatio(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()

        
        original_size = os.path.getsize(filepath)

        with gzip.GzipFile(filename='temp', mode='w', compresslevel=9) as f:
            f.write(content)

        compressed_size = os.path.getsize('temp')
        os.remove('temp')
        # c = bz2.compress(content)
        # fh = open('temp', "wb")
        # fh.write(c)
        # fh.close()
        compression_ratio = compressed_size/ original_size

        return compression_ratio

    def  get_geohash_from_path(self,p):
        return p.split("Landsat8C1L1/")[1][:5]

    def accessFiles(self, paths, isSentinel=True):
            loadedImages = []
            loaded_image = gdal.Open(paths)
            image = loaded_image.ReadAsArray() #(11, h, w)
            image = np.moveaxis(image, 0, -1)  #(h, w, 11)
            snowCover = 0
            factor = 0.0002
            if not isSentinel:
                # For Landsat 8 -> Red, Blue, Green, NIR, SWIR1, SWIR2, TIR 11, TIR 12
                image = image[:,:,[3,2,1,4,5,6,8,9]]
                image = resize(image, (self.targetW, self.targetH), preserve_range=True)  # ( 128, 128)
            else:
                # For Sentinel 2 we get 8 bands -> (Red, Blue, Green , NIR, SWIR1, SWIR2 , NDVI Maps (-1, 1), Edge Map (0,1))
                newp = paths.replace("-0.tif", "-1.tif")
                loaded_image2 = gdal.Open(newp)
                swirBand = loaded_image2.ReadAsArray()
                swirBand = np.moveaxis(swirBand, 0, -1)[:,:, [4, 5]]
                swirBand = resize(swirBand, (self.targetW, self.targetH), preserve_range=True) * factor
                # Bands: Red,Green,Blue, NIR
                newp = paths.replace("-0.tif", "-3.tif")
                rgbBand = gdal.Open(newp).ReadAsArray()
                rgbBand = np.moveaxis(rgbBand, 0, -1) / 255.0
                rgbBand = resize(rgbBand, (self.targetW, self.targetH), preserve_range=True)

                nirBand = image[:, :, 3]
                nirBand = resize(nirBand, (self.targetW, self.targetH), preserve_range=True) * factor
                image = np.dstack((rgbBand, nirBand, swirBand))
                ndvi_map = np.array((image[:,:,3] - image[:,:,0])/(image[:,:,3] + image[:,:,0])).reshape(self.targetW, self.targetH, 1)

                imageBW = rgb2gray(image[:,:,[0,1,2]])
                snowCover = self.getSnowCoverge(self.getNDSImap(image[:,:,1], image[:,:,4]), image[:,:,0])
                edges = canny(imageBW, sigma=0).reshape(self.targetW, self.targetH, 1).astype(int)

                image = np.dstack((image, ndvi_map, edges))
            loadedImages.append(image)
            return loadedImages,image, snowCover

    def getAllSentiGeo(self):
        geohashs = []
        startT, endT = self.getTime('2020-01-01', '2020-12-01')
        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=None, recurse=False,
                                             platform='Sentinel-2', max_cloud_coverage=0.3,
                                             min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        for (node, image) in listImages:
            if image.geocode not in geohashs:
                geohashs.append(image.geocode)

        return geohashs

    def getSentinelTileAtGivenTime(self, maxSentCloud=0.3):
        startT, endT = self.getTime('2020-01-01', '2020-12-01')
        metaSentinel = {}

        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=None, recurse=False,
                                             platform='Sentinel-2',  max_cloud_coverage = maxSentCloud, min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        try:
            (node, b_image) = listImages.__next__()
            while True:
                for p in b_image.files:
                    if p.path.endswith('-0.tif'):
                        pathf = p.path
                        if b_image.geocode in metaSentinel.keys():
                            old_data = metaSentinel.get(b_image.geocode)
                            old_data.append([b_image.timestamp, pathf])
                            metaSentinel[b_image.geocode] = old_data
                        else:
                            metaSentinel[b_image.geocode] = [[b_image.timestamp, pathf]]
                (_, b_image) = listImages.__next__()

        except StopIteration:
            # for g in metaSentinel:
            #      print("geohash : {}, found files: {}".format(g, len(metaSentinel.get(g))))
            return metaSentinel

    def getLansdsatTileAtGivenTime(self, maxLandCloud=0.4):
        startT, endT = self.getTime('2015-01-01', '2020-01-01')
        metaLandsat = {}

        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=None, recurse=False,
                                             platform='Landsat8C1L1',  max_cloud_coverage = 0.3, min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        try:
            (node, b_image) = listImages.__next__()
            while True:
                for p in b_image.files:
                    if p.path.endswith('0.tif'):
                        pathf = p.path
                        newK = b_image.geocode + "_" + self.convertEpochToDate(b_image.timestamp)[:4]
                        if newK in metaLandsat.keys():
                            old_data = metaLandsat.get(newK)
                            old_data.append([b_image.timestamp, pathf])
                            metaLandsat[newK] = old_data
                        else:
                            metaLandsat[newK] = [[b_image.timestamp, pathf]]
                (_, b_image) = listImages.__next__()
        except StopIteration:
            # for g in metaLandsat:
            #     print("geohash : {}, found files: {}".format(g, len(metaLandsat.get(g))))
            return metaLandsat

    def closest_sentinel(self, geohash, timestamp):

        if geohash in self.metaSentinel.keys():
            times = self.metaSentinel.get(geohash)
            allpath = []
            min_diff = np.inf
            path_min = ''
            for t in times:
                timeav = t[0]
                pathav = t[1]
                if abs(timeav - timestamp) <= 86400 * 8:
                    allpath.append(t[1])
                    if abs(timeav - timestamp) < min_diff:
                        path_min = pathav
                        min_diff = abs(timeav - timestamp)

            if path_min != '':
                _, image, snowFrac = self.accessFiles(path_min, isSentinel=True)
                if snowFrac > 0.3:
                    for p in allpath:
                        _, image, snowFrac = self.accessFiles(p, isSentinel=True)
                        if snowFrac < 0.3:
                            return image, p
                    return None, None
                return image, path_min
            else:
                return None, None
        else:
            return None, None

    def closest_landsat(self, geohash, timestamp):
            normal_time = self.convertEpochToDate(timestamp)
            years = ['2015','2016','2017', '2018', '2019']
            path_min = ''
            for y in years:
                newtimestamp = self.convertDateToEpoch(y + normal_time[4:])
                newK = geohash + "_" + y
                if newK in self.metaLandsat.keys():
                        times = self.metaLandsat.get(newK)
                        min_diff = np.inf
                        path_min = ''
                        for t in times:
                            timeav = t[0]
                            pathav = t[1]
                            if abs(timeav - newtimestamp) <= 86400 * 7 and abs(timeav - newtimestamp) < min_diff:
                                path_min = pathav
                                min_diff = abs(timeav - newtimestamp)
                        if path_min != '':
                            return path_min

            if path_min == '':
                return None

    def train_and_test_landsat_paths(self):
        '''
        This method returns paths to cloud mask and training target images paths
        :param ccThreshTest: maximum cloud coverage for training data (remove clouds for image this ccThreshTrain + 0.5 <= cc <= ccThreshTest)
        :param ccThreshTrain: maximum cloud coverage for testing data (target for images used for training cc <= ccThreshTrain)
        :return: [[path, cloud_coverage],...] array of paths to cloud mask, [path1, ...] array of paths to clean target images
        '''
        start_date='2020-05-01'
        end_date='2020-11-01'

        self.cloud_masks = {}
        stip_iter_land = stippy.list_node_images(self.host_addr, platform='Landsat8C1L1', album=self.album,
                                                 min_pixel_coverage=self.pix_cover, source='raw',
                                                 start_timestamp=self.convertDateToEpoch(start_date), geocode=None,
                                                 end_timestamp=self.convertDateToEpoch(end_date), recurse=False, max_cloud_coverage=0.80
                                                    )
        # from 0.15 to 1.0 cloud coverage
        for (node, image) in stip_iter_land:
            if image.cloudCoverage >= 0.20 and image.cloudCoverage < 0.80:
                if image.files[0].path.endswith("-0.tif"):
                    p = image.files[0].path
                    if image.geocode in self.cloud_masks.keys():
                        paths = self.cloud_masks.get(image.geocode)
                        paths.append([p,image.cloudCoverage])
                        self.cloud_masks[image.geocode] = paths
                    else:
                        self.cloud_masks[image.geocode] = [[p, image.cloudCoverage]]
                else:
                    continue

        if len(self.cloud_masks) == 0 or self.cloud_masks == {}:
            print("returning as no cloud mask found")
            return None, None, None, None, None

        if self.istrain:
            start_date = '2020-05-01'
            end_date = '2020-11-01'
        else:
            start_date='2020-04-01'
            end_date='2020-05-01'
        stip_iter_land = stippy.list_node_images(self.host_addr, platform='Landsat8C1L1', album=self.album,
                                                 min_pixel_coverage=self.pix_cover, source='raw',
                                                 start_timestamp=self.convertDateToEpoch(start_date),geocode=None,
                                                 end_timestamp=self.convertDateToEpoch(end_date), recurse=False,
                                                 max_cloud_coverage=0.10
                                                 )
        self.inp_data_dic, self.inp_timstamps = {},{}
        SentiP = self.getAllSentiGeo()

        for (node, image) in stip_iter_land:
            if image.geocode not in SentiP:
                continue
            if image.files[0].path.endswith("-0.tif"):
                p = image.files[0].path
                if image.geocode in self.inp_data_dic.keys():
                    paths = self.inp_data_dic.get(image.geocode)
                    paths.append(p)
                    self.inp_data_dic[image.geocode] = paths

                    timstampsA = self.inp_timstamps.get(image.geocode)
                    timstampsA.append(image.timestamp)
                    self.inp_timstamps[image.geocode] = timstampsA
                else:
                    self.inp_data_dic[image.geocode] = [p]
                    self.inp_timstamps[image.geocode] = [image.timestamp]           # Number of clean image for given geohash
            else:
                continue

        self.globalCloudMask = []
        for g in self.cloud_masks:
            for (pathCM, cc) in self.cloud_masks.get(g):
                self.globalCloudMask.append([pathCM, cc])
        if self.istrain:
            random.Random(44).shuffle(self.globalCloudMask)
        else:
            random.Random(100).shuffle(self.globalCloudMask)
        print("Total Cloud Mask found :", len(self.globalCloudMask))
        return

    def load_iterator_from_paths(self, resize_image = True, batch_size=None, geohashes=None):
        landsat_batch_x_cloudy, land_cloudy_with_clouds_batch,landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = [], [], [], [],[], [], []
        if self.istrain:
            with open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfo.txt", 'r') as inpFile:
                lines = [line.rstrip() for line in inpFile]
            random.Random(22).shuffle(lines)
        if not self.istrain :
            with open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfoValidation.txt",
                      'r') as inpFile:
                lines = [line.rstrip() for line in inpFile]
            random.Random(99).shuffle(lines)

        for sample in lines:
                    geo, cc, path_to_clean_image, path_to_cloud_mask, path_to_sentinel2, path_to_prev_landsat = sample.split(",")
                    if geohashes is not None:
                        if geo[3] not in geohashes:
                            continue
                    
                    _, sentI, _ = self.accessFiles(path_to_sentinel2, isSentinel=True)

                    _, landPrev, _ = self.accessFiles(path_to_prev_landsat, isSentinel=False)
                    landPrev = self.normalize11(self.scale_landsat_images(landPrev))

                    rasterL = gdal.Open(path_to_clean_image)
                    landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                    if resize_image is True:
                        landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

                    # Red, Green, Blue, NIR, TIR
                    tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4, 5, 6, 8, 9]])

                    raster = gdal.Open(path_to_cloud_mask)
                    actualCloudPixel = raster.ReadAsArray().transpose(2, 1, 0)

                    if resize_image is True:
                        actualCloudPixel = resize(actualCloudPixel, (self.targetW,self.targetH), preserve_range=True)

                    cloud_mask_array = actualCloudPixel[:, :, 10]
                    cloudm = np.empty((self.targetW, self.targetH, 8))

                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or cloud_mask_array[i][j] >= 7872:
                                cloudm[i,j] = actualCloudPixel[i,j,[3,2,1,4,5,6,8,9]]
                            else:
                                cloudm[i,j] = [0,0,0,0,0,0,0,0]

                    land_cloudy = copy.deepcopy(tmp_land)
                    land_cloudy_with_clouds = copy.deepcopy(tmp_land)

                    if self.all_black_clouds is True:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
                    cloudMask01 = np.zeros((self.targetW, self.targetH, 8))
                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if not all(cloudm[i, j] == 0) and self.all_black_clouds is False:
                                land_cloudy[i, j] = cloudm[i, j, :]

                            if not all(cloudm[i, j] == 0) and self.all_black_clouds is True:
                                land_cloudy[i, j] = [-1, -1, -1, -1, -1, -1, -1, -1]
                                land_cloudy_with_clouds[i,j] = cloudm[i, j, :]

                            if not all(cloudm[i, j] == 0):
                                cloudMask01[i, j] = [1, 1, 1, 1, 1, 1, 1, 1]

                    land_cloudy_with_clouds = self.normalize11(self.scale_landsat_images(np.array(land_cloudy_with_clouds)))
                    if self.all_black_clouds is False:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))

                    geo_batch.append(geo)
                    cc_batch.append(cc)

                    land_cloudy_with_clouds_batch.append(land_cloudy_with_clouds)
                    landsat_batch_x_cloudy.append(land_cloudy)
                    landsat_batch_y_cloud_free.append(
                        np.dstack((self.normalize11(self.scale_landsat_images(np.array(tmp_land))), np.array(cloudMask01))))

                    sent_batch.append(np.array(sentI))

                    if landPrev.shape[0] == 1:
                        landPrev = np.squeeze(landPrev, axis=0)

                    landsat_prev_batch.append(landPrev)

                    if batch_size is None:
                        batch_size = self.batch_size
                    if batch_size == len(landsat_batch_x_cloudy):
                        if self.all_black_clouds is True:
                            yield np.array(landsat_batch_x_cloudy), np.array(land_cloudy_with_clouds_batch), np.array(landsat_batch_y_cloud_free), np.array(
                            sent_batch), np.array(landsat_prev_batch), geo_batch, cc_batch
                        else:
                            yield None, np.array(land_cloudy_with_clouds_batch), np.array(
                                landsat_batch_y_cloud_free), np.array(
                                sent_batch), np.array(landsat_prev_batch), geo_batch, cc_batch
                        landsat_batch_x_cloudy, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = [], [], [], [], [], []

    def load_iterator_from_paths_with_complexity_time(self, resize_image=True, batch_size=None, geohashes=None):
        landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, timestamp_batch, landsat_prev_batch, geo_batch, cc_batch, complexity_batch = [], [], [], [], [], [], [], [], []
        if self.istrain:
            with open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfo.txt",
                      'r') as inpFile:
                lines = [line.rstrip() for line in inpFile]
            random.Random(22).shuffle(lines)
        if not self.istrain:
            with open(
                    "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfoValidation.txt",
                    'r') as inpFile:
                lines = [line.rstrip() for line in inpFile]
            random.Random(99).shuffle(lines)

        for sample in lines:
            geo, cc, path_to_clean_image, path_to_cloud_mask, path_to_sentinel2, path_to_prev_landsat = sample.split(
                ",")
            if geohashes is not None:
                if geo[3] not in geohashes:
                    continue

            _, sentI, _ = self.accessFiles(path_to_sentinel2, isSentinel=True)

            _, landPrev, _ = self.accessFiles(path_to_prev_landsat, isSentinel=False)
            landPrev = self.normalize11(self.scale_landsat_images(landPrev))

            rasterL = gdal.Open(path_to_clean_image)
            timest = self.convertEpochToDate(int(rasterL.GetMetadata('STIP')['TIMESTAMP']))

            landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
            if resize_image is True:
                landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

            # Red, Green, Blue, NIR, TIR
            tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4, 5, 6, 8, 9]])

            raster = gdal.Open(path_to_cloud_mask)
            actualCloudPixel = raster.ReadAsArray().transpose(2, 1, 0)

            if resize_image is True:
                actualCloudPixel = resize(actualCloudPixel, (self.targetW, self.targetH), preserve_range=True)

            cloud_mask_array = actualCloudPixel[:, :, 10]
            cloudm = np.empty((self.targetW, self.targetH, 8))

            for i in range(0, self.targetH):
                for j in range(0, self.targetW):
                    if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or \
                            cloud_mask_array[i][j] >= 7872:
                        cloudm[i, j] = actualCloudPixel[i, j, [3, 2, 1, 4, 5, 6, 8, 9]]
                    else:
                        cloudm[i, j] = [0, 0, 0, 0, 0, 0, 0, 0]

            land_cloudy = copy.deepcopy(tmp_land)
            land_cloudy_with_clouds = copy.deepcopy(tmp_land)

            if self.all_black_clouds is True:
                land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
            cloudMask01 = np.zeros((self.targetW, self.targetH, 8))
            for i in range(0, self.targetH):
                for j in range(0, self.targetW):
                    if not all(cloudm[i, j] == 0) and self.all_black_clouds is False:
                        land_cloudy[i, j] = cloudm[i, j, :]

                    if not all(cloudm[i, j] == 0) and self.all_black_clouds is True:
                        land_cloudy[i, j] = [-1, -1, -1, -1, -1, -1, -1, -1]
                        land_cloudy_with_clouds[i, j] = cloudm[i, j, :]

                    if not all(cloudm[i, j] == 0):
                        cloudMask01[i, j] = [1, 1, 1, 1, 1, 1, 1, 1]

            land_cloudy_with_clouds = self.normalize11(self.scale_landsat_images(np.array(land_cloudy_with_clouds)))
            if self.all_black_clouds is False:
                land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))

            geo_batch.append(geo)
            cc_batch.append(cc)
            timestamp_batch.append(timest)

            land_cloudy_with_clouds_batch.append(land_cloudy_with_clouds)
            complexity_batch.append(self.getCompressRatio(path_to_clean_image.replace("-0.tif", "-3.tif")))
            landsat_batch_x_cloudy.append(land_cloudy)
            landsat_batch_y_cloud_free.append(
                np.dstack((self.normalize11(self.scale_landsat_images(np.array(tmp_land))), np.array(cloudMask01))))

            sent_batch.append(np.array(sentI))

            if landPrev.shape[0] == 1:
                landPrev = np.squeeze(landPrev, axis=0)

            landsat_prev_batch.append(landPrev)

            if batch_size is None:
                batch_size = self.batch_size
            if batch_size == len(landsat_batch_x_cloudy):
                if self.all_black_clouds is True:
                    yield np.array(landsat_batch_x_cloudy), np.array(land_cloudy_with_clouds_batch), np.array(
                        landsat_batch_y_cloud_free), np.array(
                        sent_batch), np.array(landsat_prev_batch), geo_batch, cc_batch, complexity_batch, timestamp_batch
                else:
                    yield None, np.array(land_cloudy_with_clouds_batch), np.array(
                        landsat_batch_y_cloud_free), np.array(
                        sent_batch), np.array(landsat_prev_batch), geo_batch, cc_batch, complexity_batch,timestamp_batch
                landsat_batch_x_cloudy, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch, complexity_batch,timestamp_batch =[], [],[], [], [], [], [], []

    def load_iterator_from_paths_for_training(self, resize_image = True, batch_size=None, lossMethod='mse', geohashes=None):
        while True:
            landsat_batch_x_cloudy, land_cloudy_with_clouds_batch,landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = [], [], [], [],[], [], []
            if self.istrain:
                fileN = 'InputDatasetInfo'
            else:
                fileN = 'InputDatasetInfoValidation'

            with open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + fileN +".txt", 'r') as inpFile:
                lines = [line.rstrip() for line in inpFile]
            inpFile.close()

            random.Random(22).shuffle(lines)
            count = 0
            for sample in lines:
                        geo, cc, path_to_clean_image, path_to_cloud_mask, path_to_sentinel2, path_to_prev_landsat = sample.split(",")
                        if geohashes is not None:
                            if geo[3] not in geohashes:
                                continue

                        _, sentI, _ = self.accessFiles(path_to_sentinel2, isSentinel=True)

                        _, landPrev, _ = self.accessFiles(path_to_prev_landsat, isSentinel=False)
                        landPrev = self.normalize11(self.scale_landsat_images(landPrev))

                        rasterL = gdal.Open(path_to_clean_image)
                        landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                        if resize_image is True:
                            landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

                        # Red, Green, Blue, NIR, TIR
                        tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4, 5, 6, 8, 9]])

                        raster = gdal.Open(path_to_cloud_mask)
                        actualCloudPixel = raster.ReadAsArray().transpose(2, 1, 0)

                        if resize_image is True:
                            actualCloudPixel = resize(actualCloudPixel, (self.targetW,self.targetH), preserve_range=True)

                        cloud_mask_array = actualCloudPixel[:, :, 10]
                        cloudm = np.empty((self.targetW, self.targetH, 8))

                        for i in range(0, self.targetH):
                            for j in range(0, self.targetW):
                                if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or cloud_mask_array[i][j] >= 7872:
                                    cloudm[i,j] = actualCloudPixel[i,j,[3,2,1,4,5,6,8,9]]
                                else:
                                    cloudm[i,j] = [0,0,0,0,0,0,0,0]

                        land_cloudy = copy.deepcopy(tmp_land)
                        land_cloudy_with_clouds = copy.deepcopy(tmp_land)

                        if self.all_black_clouds is True:
                            land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
                        cloudMask01 = np.zeros((self.targetW, self.targetH, 8))
                        for i in range(0, self.targetH):
                            for j in range(0, self.targetW):
                                if not all(cloudm[i, j] == 0) and self.all_black_clouds is False:
                                    land_cloudy[i, j] = cloudm[i, j, :]

                                if not all(cloudm[i, j] == 0) and self.all_black_clouds is True:
                                    land_cloudy[i, j] = [-1, -1, -1, -1, -1, -1, -1, -1]
                                    land_cloudy_with_clouds[i,j] = cloudm[i, j, :]

                                if not all(cloudm[i, j] == 0):
                                    cloudMask01[i, j] = [1, 1, 1, 1, 1, 1, 1, 1]

                        land_cloudy_with_clouds = self.normalize11(self.scale_landsat_images(np.array(land_cloudy_with_clouds)))
                        if self.all_black_clouds is False:
                            land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))

                        geo_batch.append(geo)
                        cc_batch.append(cc)

                        land_cloudy_with_clouds_batch.append(land_cloudy_with_clouds)
                        landsat_batch_x_cloudy.append(land_cloudy)
                        # landsat_batch_y_cloud_free.append(
                        #     np.dstack((self.normalize11(self.scale_landsat_images(np.array(tmp_land))), np.array(cloudMask01))))

                        landsat_batch_y_cloud_free.append(
                            (self.normalize11(self.scale_landsat_images(np.array(tmp_land)))))

                        sent_batch.append(np.array(sentI))

                        if landPrev.shape[0] == 1:
                            landPrev = np.squeeze(landPrev, axis=0)

                        landsat_prev_batch.append(landPrev)

                        if batch_size is None:
                            batch_size = self.batch_size
                        if batch_size == len(landsat_batch_x_cloudy):
                            # if lossMethod == 'mse':
                            #     landsat_batch_y_cloud_free = landsat_batch_y_cloud_free[:, :, :, :8]
                            if self.all_black_clouds is True:
                                yield [np.array(landsat_batch_x_cloudy),  np.array(landsat_prev_batch), np.array(sent_batch)], np.array(landsat_batch_y_cloud_free)
                            else:
                                yield [np.array(land_cloudy_with_clouds_batch), np.array(landsat_prev_batch), np.array(sent_batch)], np.array(
                                    landsat_batch_y_cloud_free)
                            landsat_batch_x_cloudy, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = [], [], [], [], [], []
                            count += batch_size

    def save_input_output_files_validation(self):
        if self.cloud_masks is None or self.globalCloudMask is []:
            return
        file = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfoValidation.txt", 'w')
        globalCountCloudMask, globalCleanImage = 0,0

        for geo in self.inp_data_dic.keys():
            clean_image_count = 0
            inp_data = self.inp_data_dic.get(geo)
            random.Random(22).shuffle(inp_data)
            for i in range(len(inp_data)):
                if clean_image_count == 1 :
                    break
                path_to_clean_image = inp_data[i]
                timestamp = self.inp_timstamps.get(geo)[i]

                _, path_to_sentinel2 = self.closest_sentinel(geo, timestamp)
                path_to_prev_landsat = self.closest_landsat(geo, timestamp)
                if path_to_sentinel2 is '' or path_to_sentinel2 is None or path_to_prev_landsat is '' or path_to_prev_landsat is None:
                        continue

                clean_image_count += 1
                try:
                        path_to_cloud_mask, cc = self.globalCloudMask[globalCountCloudMask]
                except IndexError:
                        globalCountCloudMask = 0
                        path_to_cloud_mask,cc = self.globalCloudMask[globalCountCloudMask]

                globalCountCloudMask += 1
                globalCleanImage += 1
                file.write(geo + ',' + str(cc) + ',' + path_to_clean_image + ',' + path_to_cloud_mask + ',' +
                               path_to_sentinel2 + ',' + path_to_prev_landsat + '\n')


        file.close()
        file1 = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/TotalCleanImagesValidation.txt", "w")
        file1.write(str(globalCleanImage))
        file1.close()

    def save_input_output_files(self):
        if self.cloud_masks is None or self.globalCloudMask is []:
            return

        file = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/InputDatasetInfo.txt", 'w')
        globalCountCloudMask, clean_image_count = 0, 0
        for geo in self.inp_data_dic.keys():
            clean_image_count += len(self.inp_data_dic.get(geo))
            inp_data = self.inp_data_dic.get(geo)
            for i in range(0,len(inp_data)):
                path_to_clean_image = inp_data[i]
                timestamp = self.inp_timstamps.get(geo)[i]

                _, path_to_sentinel2 = self.closest_sentinel(geo, timestamp)
                path_to_prev_landsat = self.closest_landsat(geo, timestamp)

                if path_to_sentinel2 is '' or path_to_sentinel2 is None or path_to_prev_landsat is '' or path_to_prev_landsat is None:
                    clean_image_count -= 1
                    continue
                try:
                    path_to_cloud_mask, cc = self.globalCloudMask[globalCountCloudMask]
                except IndexError:
                    globalCountCloudMask = 0
                    path_to_cloud_mask,cc = self.globalCloudMask[globalCountCloudMask]

                globalCountCloudMask += 1
                file.write(geo + ',' + str(cc) + ',' + path_to_clean_image + ',' + path_to_cloud_mask + ',' +
                           path_to_sentinel2 + ',' + path_to_prev_landsat + '\n')

        file.close()
        file1 = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/TotalCleanImages.txt", "w")
        file1.write(str(clean_image_count))
        file1.close()


    def getNDVIMapLand(self, landsatI):
        landsatI = self.normalize01(self.scale_landsat_images(landsatI))
        ndvi = (landsatI[:, :,3] - landsatI[:, :, 0]) / (landsatI[:, :, 3] + landsatI[:, :,0])
        ndvi = np.array(ndvi).reshape((self.targetH, self.targetW, 1))
        return ndvi


    def display_training_images(self, epoch,
                                    landsat_batch_x_cloudy,land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch,cc_batch, is_train=True):
            output_dir = './sen/'
            cloudy_img_land = self.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)
            land_cloudy_with_clouds = self.denormalize11(land_cloudy_with_clouds_batch[-1]).astype(np.uint8)
            target_cloudfree_land = self.denormalize11(landsat_batch_y_cloud_free[-1]).astype(np.uint8)
            inp_sent_img = sent_batch[-1]
            inp_prev_landsat = np.array([self.denormalize11(img).astype(np.uint8) for img in landsat_prev_batch])
            titles = ['Previous Year Landsat', 'Sentinel-2 RGB', 'Sentinel-2 Edges', 'Input Cloudy', 'Input Cloud Mask', 'Target Cloud Free']

            r, c = 2, 3
            fig, axarr = plt.subplots(r, c,  figsize=(15, 12))
            np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

            axarr[0,0].imshow(inp_prev_landsat[0][:,:,:3])
            axarr[0,0].set_title(titles[0], fontdict={'fontsize':15})

            axarr[0,1].imshow(inp_sent_img[:,:,:3])
            axarr[0,1].set_title(titles[1], fontdict={'fontsize': 15})

            axarr[0,2].imshow(inp_sent_img[:, :, -1])
            axarr[0,2].set_title(titles[2], fontdict={'fontsize': 15})

            axarr[1,0].imshow(land_cloudy_with_clouds[:,:,:3])
            axarr[1,0].set_title(titles[3], fontdict={'fontsize': 15})

            axarr[1,1].imshow(cloudy_img_land[:,:,:3])
            axarr[1,1].set_title(titles[4],  fontdict={'fontsize':15})

            axarr[1,2].imshow(target_cloudfree_land[:,:,:3])
            axarr[1,2].set_title(titles[5], fontdict={'fontsize': 15})
            plt.suptitle("Geohash: {} Cloud Coverage: {}%".format(geo_batch[-1], str(round(float(cc_batch[-1]) * 100, 3)), fontsize=30))
            fig.savefig(output_dir + "%s.png" % (epoch))
            plt.close()

if __name__ == '__main__':



    dataH = DatasetHandling(w = 128, h = 128, no_of_timesteps=1,album='iowa-2015-2020-spa', batch_size=1,
                            istrain=True, all_black_clouds = True, folderI = 1001, saveInputMetaData = False)

    path = "/s/lattice-178/a/nobackup/galileo/stip-images/iowa-2015-2020-spa/Landsat8C1L1/9zmck/raw/LC08_L1TP_026032_20200523_20200607_01_T1-0.tif"
    pp = gdal.Open(path)
    actualCloudPixel = pp.ReadAsArray().transpose(1, 2, 0)
    land_cloudy_with_clouds = dataH.normalize01(dataH.scale_landsat_images(np.array(actualCloudPixel[:, :, [3, 2, 1]])))
    plt.imshow(land_cloudy_with_clouds)
    print("Sving")
    plt.savefig("./cloudmsimage2.png")
    plt.close()
    print("saved")
    exit(0)

    it = dataH.load_iterator_from_paths_with_complexity_time(resize_image=True, batch_size=1,geohashes=None)
    countI = 0
    while True:
         try:
             inp, landsat_batch_y_cloud_free = it.__next__()
             landsat_batch_x_cloudy,landsat_prev_batch, sent_batch = inp
             landsat_batch_x_cloudy = dataH.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

             countI += 1
             print(np.min(landsat_batch_x_cloudy[:,:,0]))
             print(np.max(landsat_batch_x_cloudy[:, :,0]))
             # dataH.display_training_images(random.randint(0,400), landsat_batch_x_cloudy,land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch,cc_batch)
             print(countI)
             # print(cloudy_img_land.shape, target_cloudfree_land.shape, prev_sent.shape, inp_prev_landsat.shape, cloudMask01Batch.shape)
         except StopIteration:
             print("Got total training samples: " , countI)
             break
