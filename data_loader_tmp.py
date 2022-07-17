import os
import sys
from skimage.color import rgb2gray
import random
import gdal
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir + '/../')
import stippy
from time import strptime, mktime
import datetime
import socket
import numpy as np
import sys
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


class DatasetHandling():

    def __init__(self, w=128, h=128, no_of_timesteps=1,
                 pix_cover=1.0,cloud_cov=0.3, album='iowa-2015-2020-spa', batch_size=1, istrain=True):

        self.targetH = h
        self.targetW = w
        self.pix_cover = pix_cover
        self.cloud_cov = cloud_cov
        self.no_of_timesteps = no_of_timesteps
        self.album = album
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        self.metaSentinel = self.getSentinelTileAtGivenTime()
        self.metaLandsat = self.getLansdsatTileAtGivenTime()
        self.ccThreshTest = 0.9
        self.ccThreshTrain = 0.2
        self.batch_size = batch_size
        self.lenarr=[]
        self.istrain = istrain
        self.train_and_test_landsat_paths(resize_image=True)
        # self.train_geohashes = makeLandsatMetaData()
        # self.len_train_geohash = len(list(self.train_geohashes))

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


    def get_geohash_from_path(self,p):
        return p.split("Landsat8C1L1/")[1][:5]

    def accessFiles(self, paths, isSentinel=True):
            loadedImages = []
        # for p in paths:
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
                swirBand = resize(swirBand, (self.targetW, self.targetH), preserve_range=True)
                # Bands: Red,Green,Blue, NIR
                tmp_image = image[:, :, [0,1,2,3]]
                tmp_image = resize(tmp_image, (self.targetW, self.targetH), preserve_range=True)
                image = np.dstack((tmp_image, swirBand)) * factor
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
                            return image
                    return None

                # image = self.normalize11(self.scale_landsat_images(image))
                return image
            else:
                return None
        else:
            return None

    def closest_landsat(self, geohash, timestamp):
            landsat_images_prev = []
            normal_time = self.convertEpochToDate(timestamp)
            years = ['2015','2016','2017', '2018', '2019']

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
                            _,im,_ = self.accessFiles(path_min, isSentinel=False)
                            im = self.normalize11(self.scale_landsat_images(im))
                            landsat_images_prev.append(im)
                if len(landsat_images_prev) == self.no_of_timesteps:
                    break

            if landsat_images_prev == [] or len(landsat_images_prev) < self.no_of_timesteps:
                return None

            self.lenarr.append(len(landsat_images_prev))
            return np.array(landsat_images_prev)


    def train_and_test_landsat_paths(self, resize_image=True):
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
            # rn = random.random()
            if image.cloudCoverage >= 0.20 and image.cloudCoverage < 0.70:
                if image.files[0].path.endswith("-0.tif"):
                    p = image.files[0].path
                    if image.geocode in self.cloud_masks.keys():
                        paths = self.cloud_masks.get(image.geocode)
                        # if len(paths) > 10:
                        #     continue
                        paths.append([p,image.cloudCoverage])
                        self.cloud_masks[image.geocode] = paths
                    else:
                        self.cloud_masks[image.geocode] = [[p, image.cloudCoverage]]
                else:
                    continue
        if len(self.cloud_masks) == 0 or self.cloud_masks == {}:
            print("returning as no cloud mask found")
            return None, None, None, None, None

        for key in self.cloud_masks.keys():
            val = self.cloud_masks.get(key)
            random.shuffle(val)
            self.cloud_masks[key] = val

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
                                                 max_cloud_coverage=0.09
                                                 )
        self.inp_data_dic, self.inp_timstamps = {},{}
        SentiP = self.getAllSentiGeo()
        # with open('./cloudmasksPathsTest.pkl', 'wb') as f:
        #     pickle.dump(cloud_masks, f, pickle.HIGHEST_PROTOCOL)

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
        # with open('./targetImagesPathsTest.pkl', 'wb') as f2:
        #     pickle.dump(input_imgs, f2, pickle.HIGHEST_PROTOCOL)

        self.x_cloud_dic, g_x_dic = {},{}
        for g in self.cloud_masks:
            x_cloud, g_x = [], []
            for (pathCM, cc) in self.cloud_masks.get(g):
                raster = gdal.Open(pathCM)
                actualCloudPixel = raster.ReadAsArray().transpose(2, 1, 0)

                if resize_image is True:
                    actualCloudPixel = resize(actualCloudPixel, (self.targetW,self.targetH), preserve_range=True)

                cloud_mask_array = actualCloudPixel[:, :, 10]
                cm = np.empty((self.targetW, self.targetH, 8))

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        # set ice/snow pixels as 0
                        if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or cloud_mask_array[i][j] >= 7872:
                                 # or (cloud_mask_array[i][j] == 6896) or (cloud_mask_array[i][j] == 6900) or (cloud_mask_array[i][j] == 6904) or (cloud_mask_array[i][j] == 6908) :
                            cm[i,j] = actualCloudPixel[i,j,[3,2,1,4,5,6,8,9]]
                        else:
                            cm[i,j] = [0,0,0,0,0,0,0,0]

                x_cloud.append(cm)
                g_x.append(float(cc))

            self.x_cloud_dic[g] = x_cloud
            g_x_dic[g] = g_x


        return

    def getNDVIMapLand(self, landsatI):
        landsatI = self.normalize01(self.scale_landsat_images(landsatI))
        ndvi = (landsatI[:, :,3] - landsatI[:, :, 0]) / (landsatI[:, :, 3] + landsatI[:, :,0])
        ndvi = np.array(ndvi).reshape((self.targetH, self.targetW, 1))
        return ndvi


    def load_landsat_images(self, resize_image=True, batch_size=None, all_black_clouds=False):

        if self.cloud_masks is None or self.inp_data_dic is None or self.x_cloud_dic is None:
            return

        landsat_batch_x_cloudy, landsat_batch_y_cloud_free,sent_batch, landsat_prev_batch = [],[],[],[]

        for geo in self.inp_data_dic.keys():

            if geo not in self.x_cloud_dic.keys():
                # print("error : no cloud mask found for input image")
                continue
            inp_data = self.inp_data_dic.get(geo)
            for i in range(0,len(inp_data)):
                p = inp_data[i]
                timestamp = self.inp_timstamps.get(geo)[i]

                sentI = self.closest_sentinel(geo, timestamp)
                landPrev = self.closest_landsat(geo, timestamp)

                if sentI is None or landPrev is None:
                    continue

                rasterL = gdal.Open(p)
                landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                if resize_image is True:
                    landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

                # Red, Green, Blue, NIR, TIR
                tmp_land = (landsat_image_array[:, :, [3,2,1,4,5,6,8,9]])
                # tmp_land = self.scale_images(np.array(tmp_land))
                for (index, cloudm) in enumerate(self.x_cloud_dic.get(geo)):
                    land_cloudy = copy.deepcopy(tmp_land)
                    if all_black_clouds is True:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
                    cloudMask01 = np.zeros((self.targetW, self.targetH, 8))
                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if not all(cloudm[i,j] == 0) and all_black_clouds is False:
                                land_cloudy[i,j] = cloudm[i,j,:]

                            if not all(cloudm[i,j] == 0) and all_black_clouds is True:
                                land_cloudy[i,j] = [-1,-1,-1,-1,-1,-1,-1,-1]

                            if not all(cloudm[i, j] == 0):
                                cloudMask01[i, j] = [1,1,1,1,1,1,1,1]

                    if all_black_clouds is False:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))

                    landsat_batch_x_cloudy.append(land_cloudy)
                    landsat_batch_y_cloud_free.append(np.dstack((self.normalize11(self.scale_landsat_images(np.array(tmp_land))),np.array(cloudMask01))))

                    sent_batch.append(np.array(sentI))

                    if landPrev.shape[0] == 1:
                        landPrev = np.squeeze(landPrev, axis=0)

                    landsat_prev_batch.append(landPrev)

                    if batch_size is None:
                        batch_size = self.batch_size
                    if batch_size == len(landsat_batch_x_cloudy):
                        yield np.array(landsat_batch_x_cloudy), np.array(landsat_batch_y_cloud_free),np.array(sent_batch),np.array(landsat_prev_batch), geo
                        landsat_batch_x_cloudy, landsat_batch_y_cloud_free,sent_batch,landsat_prev_batch  = [],[],[],[]



    def load_landsat_images_random(self, resize_image=True, batch_size=None, all_black_clouds=False):

        if self.cloud_masks is None or self.inp_data_dic is None or self.x_cloud_dic is None:
            return

        landsat_batch_x_cloudy, landsat_batch_y_cloud_free,sent_batch, landsat_prev_batch = [],[],[],[]

        for geo in self.inp_data_dic.keys():

            if geo not in self.x_cloud_dic.keys():
                # print("error : no cloud mask found for input image")
                continue
            inp_data = self.inp_data_dic.get(geo)
            for i in range(0,len(inp_data)):
                p = inp_data[i]
                timestamp = self.inp_timstamps.get(geo)[i]

                sentI = self.closest_sentinel(geo, timestamp)
                landPrev = self.closest_landsat(geo, timestamp)

                if sentI is None or landPrev is None:
                    continue

                rasterL = gdal.Open(p)
                landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                if resize_image is True:
                    landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

                # Red, Green, Blue, NIR, TIR
                tmp_land = (landsat_image_array[:, :, [3,2,1,4,5,6,8,9]])
                # tmp_land = self.scale_images(np.array(tmp_land))
                for (index, cloudm) in enumerate(self.x_cloud_dic.get(geo)):
                    land_cloudy = copy.deepcopy(tmp_land)
                    if all_black_clouds is True:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
                    cloudMask01 = np.zeros((self.targetW, self.targetH, 8))
                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if not all(cloudm[i,j] == 0) and all_black_clouds is False:
                                land_cloudy[i,j] = cloudm[i,j,:]

                            if not all(cloudm[i,j] == 0) and all_black_clouds is True:
                                land_cloudy[i,j] = [-1,-1,-1,-1,-1,-1,-1,-1]

                            if not all(cloudm[i, j] == 0):
                                cloudMask01[i, j] = [1,1,1,1,1,1,1,1]

                    if all_black_clouds is False:
                        land_cloudy = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))

                    landsat_batch_x_cloudy.append(land_cloudy)
                    landsat_batch_y_cloud_free.append(np.dstack((self.normalize11(self.scale_landsat_images(np.array(tmp_land))),np.array(cloudMask01))))

                    sent_batch.append(np.array(sentI))

                    if landPrev.shape[0] == 1:
                        landPrev = np.squeeze(landPrev, axis=0)

                    landsat_prev_batch.append(landPrev)

                    if batch_size is None:
                        batch_size = self.batch_size
                    if batch_size == len(landsat_batch_x_cloudy):
                        yield np.array(landsat_batch_x_cloudy), np.array(landsat_batch_y_cloud_free),np.array(sent_batch),np.array(landsat_prev_batch), geo
                        landsat_batch_x_cloudy, landsat_batch_y_cloud_free,sent_batch,landsat_prev_batch  = [],[],[],[]



    def display_training_images(self, epoch,
                                    cloudy_img_land, target_cloudfree_land, inp_sent_img, inp_prev_landsat, geo, is_train=True):

            output_dir = './sen/'

            r, c = 2, 4


            # fakeImg = (self.dataloader.denormalize11(self.generator.predict([cloudy_img_land,inp_prev_landsat, inp_sent_img])[-1])).astype(np.uint8)
            cloudy_img_land = self.denormalize11(cloudy_img_land[-1]).astype(np.uint8)
            target_cloudfree_land = self.denormalize11(target_cloudfree_land[-1]).astype(np.uint8)
            inp_sent_img = inp_sent_img[-1]
            inp_prev_landsat = np.array([self.denormalize11(img).astype(np.uint8) for img in inp_prev_landsat])
            print("inp prev hape ", inp_prev_landsat.shape)

            titles4 = ['Landsat Y1', 'Landsat Y2', 'Landsat Y3', 'Landsat Y4',  'Sentinel-2', 'Input cloudy Landsat', 'Target', 'Predicted']
            titles3 = ['Landsat Y1', 'Landsat Y2', 'Landsat Y3', 'Sentinel-2', 'Input cloudy Landsat', 'Target','Predicted']
            titles2 = ['Landsat Y1', 'Landsat Y2', 'Sentinel-2', 'Input cloudy Landsat', 'Target','Predicted']
            titles1 = ['Landsat Y1', 'Sentinel-2-RGB', 'Sen2 NDVI', 'Sen2 Edge', 'Input cloudy Landsat',  'Target', 'cloudmask']


            titles = titles1


            fig, axarr = plt.subplots(r, 4,  figsize=(15, 12))
            np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

            for col in range(1):

                    axarr[0, col].imshow(inp_prev_landsat[col][:,:,:3])

                    axarr[0, col].set_title(titles[col], fontdict={'fontsize':15})

            axarr[0,1].imshow(inp_sent_img[:,:,:3])
            axarr[0,1].set_title(titles[1], fontdict={'fontsize': 15})

            axarr[0, 2].imshow(inp_sent_img[:, :, 3], cmap=plt.cm.summer)
            axarr[0, 2].set_title(titles[2], fontdict={'fontsize': 15})

            axarr[0, 3].imshow(inp_sent_img[:, :, -1])
            axarr[0, 3].set_title(titles[3], fontdict={'fontsize': 15})

            axarr[r-1, 0].imshow(cloudy_img_land[:,:,:3])
            axarr[r-1, 0].set_title(titles[4], fontdict={'fontsize': 15})

            axarr[r-1, 1].imshow(target_cloudfree_land[:,:,:3])
            axarr[r-1, 1].set_title(titles[5],  fontdict={'fontsize':15})

            axarr[r-1, 2].imshow(target_cloudfree_land[:, :, -2])
            axarr[r-1, 2].set_title('lst landsat', fontdict={'fontsize': 15})

            # axarr[r-1, 3].imshow(cloudMask01Batch[-1][:,:,0])
            # axarr[r-1, 3].set_title('Cloud Mask', fontdict={'fontsize': 15})


            fig.savefig(output_dir + "%s.png" % (epoch))
            plt.close()

def tmp():
    glob_value = {11:0,12:0, 21:0, 22:0, 23:0,24:0, 31:0, 41:0, 42:0,43:0,51:0, 52:0, 71:0, 72:0, 73:0, 74: 0, 81:0, 82:0, 90:0, 95:0}
    globArray = []
    host_add = socket.gethostbyname(socket.gethostname()) + ':15606'
    stip_iter_land = stippy.list_node_images(host_add, platform='NLCD', album='iowa-2015-2020-spa',
                                             geocode=None, source='raw',
                                             recurse=False,
                                             )

    for (node, image) in stip_iter_land:
        path = image.files[0].path
        inp_raster = gdal.Open(path)
        im = inp_raster.ReadAsArray().flatten()
        globArray.extend(im)
        val, counts = np.unique(im, return_counts=True)
        for i in range(len(val)):
            current_count = glob_value.get(val[i])
            current_count += counts[i]
            glob_value[val[i]] = current_count

    # print(glob_value)
    total_pixels = sum(glob_value.values())
    val2 , count2 = np.unique(globArray, return_counts=True)
    frac = []
    for k, v in glob_value.items():
        pct = v * 100.0 / total_pixels
        if pct > 0.0:
            frac.append(pct)
        # print(k, pct)
    for v in count2:
        pct = v * 100.0 / total_pixels
        print(pct)

    N_points = 100000
    n_bins = 15
    x = np.random.randn(N_points)
    fig, axs = plt.subplots(1, 2, tight_layout=True)

    axs[0].hist(globArray, bins=n_bins)

    axs[1].hist(globArray, bins=n_bins, density=True)

    # Now we format the y-axis to display percentage
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.savefig("./hist.png")
    plt.close()
    # print(sorted(frac, reverse=True))



if __name__ == '__main__':
    # tmp()
    dataH = DatasetHandling(128, 128, istrain=True,
                         album='iowa-2015-2020', no_of_timesteps=1, batch_size=1)
    # dataH.train_and_test_landsat_paths()

    # it = dataH.load_landsat_images(all_black_clouds=True)  # Model 1
    test_itr = dataH.load_landsat_images(all_black_clouds=False, batch_size=1)

    for i in range(10000):
        cloudy_img_land1, target_cloudfree_land1, inp_prev_sent1, inp_prev_landsat1, geo1 = test_itr.__next__()
        dataH.display_training_images(i, cloudy_img_land1, target_cloudfree_land1, inp_prev_sent1,
                                      inp_prev_landsat1, geo1)
    exit(0)

    countI = 0
    while True:
         try:
             cloudy_img_land, target_cloudfree_land, prev_sent, inp_prev_landsat, geo = it.__next__()
             countI += 1
             dataH.display_training_images(random.randint(0,1000), cloudy_img_land, target_cloudfree_land, prev_sent, inp_prev_landsat, geo)
             print(countI)
             # print(cloudy_img_land.shape, target_cloudfree_land.shape, prev_sent.shape, inp_prev_landsat.shape, cloudMask01Batch.shape)
             #
             # if geo in countI.keys():
             #     newc = countI.get(geo)
             #     countI[geo] = newc + 1
             # else:
             #     countI[geo] = 1
         except StopIteration:
             print("got :" , countI)
             break
    #
