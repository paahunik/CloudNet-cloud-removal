import pickle
import random
from time import strptime, mktime
import datetime
import socket
import numpy as np
import sys
from skimage import exposure
import cv2
import csv
np.set_printoptions(threshold=sys.maxsize)
import gdal
from skimage.transform import resize
import copy
import os
from sklearn.model_selection import train_test_split



class DatasetHandling():

    def __init__(self, w=128, h=128, data_type='train', no_of_timesteps=2,
                 pix_cover=1.0, startT='2015-01-01', endT='2020-01-01',
                 cloud_cov=0.3, album='iowa-2015-2020', batch_size=5, geoDirLimit=50, cloud_mask_limit=5, istrain=True,cluster_size=1,node_rank=0):

        self.targetH = h
        self.targetW = w
        self.data_type=data_type
        self.startT = startT
        self.endT = endT
        self.pix_cover = pix_cover
        self.cloud_cov = cloud_cov
        self.no_of_timesteps = no_of_timesteps
        self.album = album
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        #self.metaSentinel = self.getSentinelTileAtGivenTime()
        #self.metaLandsat = self.getLansdsatTileAtGivenTime()
        self.ccThreshTest = 0.9
        self.ccThreshTrain = 0.2
        self.batch_size = batch_size
        self.geoDirLimit = geoDirLimit
        self.cloud_mask_limit = cloud_mask_limit
        self.cluster_size=cluster_size
        self.node_rank=node_rank
        self.lenarr=[]
        self.istrain = istrain
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

    def scale_landsat_images(self, imgs):
        img_tmp = exposure.rescale_intensity(imgs,
                                             out_range=(0, 65535)).astype(np.int)
        img_tmp = np.sqrt(img_tmp).astype(np.int)
        return img_tmp

    def get_geohash_from_path(self,p):
        return p.split("Landsat8C1L1/")[1][:5]

    def accessFiles(self, paths, isSentinel=True):
            loadedImages = []
        # for p in paths:
            loaded_image = gdal.Open(paths)
            image = loaded_image.ReadAsArray()
            image = np.moveaxis(image, 0, -1)
            factor = 2 / 1e4

            if not isSentinel:
                image = image[:,:,[3,2,1,4,8]]
            else:
                image = image * factor

            image = resize(image, (self.targetW, self.targetH), preserve_range=True)
            # image = self.scale_images_11(image)
            loadedImages.append(image)
            return loadedImages,image

    def  getSentinelTileAtGivenTime(self, maxSentCloud=0.3):
        startT, endT = self.getTime('2020-01-01', '2021-01-01')
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
            #     print("geohash : {}, found files: {}".format(g, len(metaSentinel.get(g))))
            return metaSentinel

    def getLansdsatTileAtGivenTime(self, maxLandCloud=0.3):
        startT, endT = self.getTime('2015-01-01', '2020-01-01')
        metaLandsat = {}

        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=None, recurse=False,
                                             platform='Landsat8C1L1',  max_cloud_coverage = maxLandCloud, min_pixel_coverage=0.9,
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

            min_diff = np.inf
            path_min = ''
            for t in times:
                timeav = t[0]
                pathav = t[1]
                if abs(timeav - timestamp) <= 86400 * 7 and abs(timeav - timestamp) < min_diff:
                    path_min = pathav
                    min_diff = abs(timeav - timestamp)

            if path_min != '':
                _, image = self.accessFiles(path_min, isSentinel=True)
                # image = self.normalize11(self.scale_landsat_images(image))
                return image
            else:
                return None
        else:
            return None

    def closest_landsat(self, geohash, timestamp):
            landsat_images_prev = []
            normal_time = self.convertEpochToDate(timestamp)
            years = ['2015', '2016', '2017', '2018', '2019']

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
                            _,im = self.accessFiles(path_min, isSentinel=False)
                            im = self.normalize11(self.scale_landsat_images(im))
                            landsat_images_prev.append(im)
                if len(landsat_images_prev) == self.no_of_timesteps:
                    break

            if landsat_images_prev == [] or len(landsat_images_prev) < self.no_of_timesteps:
                return None

            self.lenarr.append(len(landsat_images_prev))
            return np.array(landsat_images_prev)


        
    def train_and_test_landsat_paths(self):
        '''
        This method returns paths to cloud mask and training target images paths
        :param ccThreshTest: maximum cloud coverage for training data (remove clouds for image this ccThreshTrain + 0.5 <= cc <= ccThreshTest)
        :param ccThreshTrain: maximum cloud coverage for testing data (target for images used for training cc <= ccThreshTrain)
        :return: [[path, cloud_coverage],...] array of paths to cloud mask, [path1, ...] array of paths to clean target images
        '''
        
        path = '/s/chopin/f/proj/fineET/landsat8' #will take as arguemnet later
        
        ################# Reading and spliting all files under provided director ############
        allGeoDirs = os.listdir(path)
        all_file_lists = []   # it will hold the all fileNames with full directory from all geohashes
        cloud_masks = []
        timestamps = []
        #print(allGeoDirs)
        limit = 0   # how many geodirs should be read 
        #### distribute geodirs to read by each machine node
        per_node_geoDirs = int(len(allGeoDirs)/self.cluster_size)
        n=per_node_geoDirs
        GeoDirs = allGeoDirs[n*self.node_rank:n*self.node_rank+n]
        print("\n***from dataloader2 hvd size and rank ",self.cluster_size,self.node_rank)
        print("*****\n total geodirs to process:  ",len(GeoDirs))
       
        for d in GeoDirs:
            limit+=1
            if limit>self.geoDirLimit:
              break
            geoPath = os.path.join(path,d) # /s/...../yxxx
            if os.path.isdir(geoPath):
               #print(geoPath)
               if not os.access(geoPath, os.R_OK):
                  continue
               allfiles = os.listdir(geoPath)
               #print(allfiles)
               for f in allfiles:
                   try:
                       if f.endswith("-0.tif"):
                          f_path = os.path.join(geoPath,f)
                          img = None
                          if not os.access(f_path,os.R_OK):
                             continue
                          img = gdal.Open(f_path)
                          
                      
                          if img != None and 'CLOUD_COVERAGE' not in img.GetMetadata('STIP'):   # some file seems not have cloud_coverage key
                              continue  
                          cloud_cov = float(img.GetMetadata('STIP')['CLOUD_COVERAGE'])
                          pix_cover = float(img.GetMetadata('STIP')['PIXEL_COVERAGE'])
                          print("cloud_coverage: {} and Pixel_coverage:{} ".format(cloud_cov,pix_cover))
                          if cloud_cov >0.4 and cloud_cov<0.8 and pix_cover>=self.pix_cover:   ### reading metadata  
                          
                             cloud_masks.append([f_path,cloud_cov])
                             
                          if cloud_cov <0.1 and pix_cover>=self.pix_cover:   ### reading metadata  
                             all_file_lists.append(f_path)  # /s/..../yxxx/123xxxx-0.tif
                             timestamps.append(img.GetMetadata('STIP')['TIMESTAMP'])
                   except StopIteration:
                       continue
                      
        ############ Divided All file Names with full path into two parts: train and test set######
        print("***** total valid files****",len(all_file_lists))
        random.seed(42)
        train_fileNames, test_fileNames = train_test_split( all_file_lists, test_size=0.4,train_size=0.6, random_state=42)
        
        if len(train_fileNames)>60:
           train_fileNames = random.sample(train_fileNames,60)
        
        self.cloud_mask_limit = 5
        if len(cloud_masks)>self.cloud_mask_limit:
           cloud_masks = random.sample(cloud_masks,self.cloud_mask_limit)
                       
        
        
        if self.istrain:
            start_date='2020-01-01'
            end_date='2020-09-01'
        else:
            start_date='2020-09-01'
            end_date='2021-01-01'

      

        # print("Total Target Images found: ", len(input_imgs))
        print("Total Cloud Masks found: ", len(cloud_masks))
        print("\nTotal Train Files : ",len(train_fileNames))

        #return cloud_masks, input_imgs, timstamps
        return cloud_masks,train_fileNames, test_fileNames,timestamps

    def train_file_and_mask_path_read(self):
        inputPath ='/s/chopin/f/proj/fineET/inputFiles'
        train_filePath = inputPath + "/train-part-"+str(self.node_rank)+".csv"
        train_maskPath = inputPath + "/train-mask-part-"+str(self.node_rank)+".csv"
    

        with open(train_filePath, newline='') as f:
            reader = csv.reader(f)
            fileNames = list(reader)
            
        train_fileNames = []
        for sublist in fileNames:
            for item in sublist: 
                train_fileNames.append(item)
            
        with open(train_maskPath, newline='') as fm:
            reader = csv.reader(fm)
            maskPaths = list(reader)
       
        cloud_maskPaths = []
        for sublist in maskPaths:
            for item in sublist: 
                cloud_maskPaths.append(item)

        #return cloud_maskPaths,train_fileNames
        return maskPaths,fileNames


    def read_pickle_data(self,file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def load_landsat_images(self, resize_image=True, batch_size=None):
        #cloud_masks, inp_data, inp_timstamps = self.train_and_test_landsat_paths()
        
        cloud_masks,train_fileNames = self.train_file_and_mask_path_read()
        #print(cloud_masks)
        
        #if self.data_type =='train':
        while True:
              inp_data = train_fileNames
              #else:
               ##   inp_data = test_fileNames
                  
              print("\nTotal Input Files : ",len(inp_data))
              print("\n ********* First input file name ", inp_data[0])
      
              x_cloud, g_x, landsat_batch_x_cloudy, landsat_batch_y_cloud_free, g_batch_x,sent_batch, landsat_prev_batch = [],[],[],[],[],[],[]
              shuffler = np.random.permutation(len(cloud_masks))
              cloud_masks = np.array(cloud_masks)[shuffler]
              for (pathCM, cc) in cloud_masks:
                  print(pathCM)
                  print(cc)
                  raster = gdal.Open(pathCM)
                  
                  print(" Line-338 cm shape {}",raster.ReadAsArray().shape)
                  
                  ############## Transpose dimension are different 
                  actualCloudPixel = raster.ReadAsArray().transpose(1, 2, 0)  # (2,1,0)
                  
                  #print(" linne-342 after transpose cm shape {}",actualCloudPixel.shape)
      
                  if resize_image is True:
                      actualCloudPixel = resize(actualCloudPixel, (self.targetW,self.targetH), preserve_range=True)
                  
                  if actualCloudPixel.shape[2]<11:   # some file don't have eleven bands 
                     continue
                  cloud_mask_array = actualCloudPixel[:, :, 10]
                 #print(" cm 10th band shape {}",cloud_mask_array.shape)
                  #print(cloud_mask_array)
                  
                  cm = np.empty((self.targetW,self.targetH, 9))
      
                  for i in range(0, self.targetH):
                      for j in range(0, self.targetW):
                          # set ice/snow pixels as 0
                          #  (array[i][j] == 6896) or (array[i][j] == 6900) or (array[i][j] == 6904) or (array[i][j] == 6908))
                          if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or cloud_mask_array[i][j] >= 7872\
                                   or (cloud_mask_array[i][j] == 6896) or (cloud_mask_array[i][j] == 6900) or (cloud_mask_array[i][j] == 6904) or (cloud_mask_array[i][j] == 6908) :
                              cm[i,j] = actualCloudPixel[i,j,[3,2,1,4,5,6,8,9,10]]  # before [3,2,1,4,8]
                          else:
                              cm[i,j] = [0,0,0,0,0,0,0,0,0]
                  x_cloud.append(cm)
                  g_x.append(float(cc))
                  #print("Cloud Mask : {}\n",cm.shape)
                  #print(cm)
                  #print("Cloud Coverage: {}\n",cc)
                  #print(cc)
      
              for i in range(0,len(inp_data)):
                  p = inp_data[i][0]
                  #timestamp = inp_timstamps[i]
                  rasterL = gdal.Open(p)
                  landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                  
                  #print("line-377 landsat image shape: {}",landsat_image_array.shape)
                  
                  if resize_image is True:
                      landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)
      
                  # Red, Green, Blue, NIR, TIR
                  if landsat_image_array.shape[2]<11:
                     continue
                  tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4,5,6,8,9,10]])
                  # tmp_land = self.scale_images(np.array(tmp_land))
      
                  for (index, cloudm) in enumerate(x_cloud):
                      land_cloudy = copy.deepcopy(tmp_land)
      
                      for i in range(0, self.targetH):
                          for j in range(0, self.targetW):
      
                              if not all(cloudm[i,j] == 0):
                                  land_cloudy[i,j] = cloudm[i,j,:]
                                  #print("line 397 shapes {} {}",land_cloudy.shape,cloudm.shape)
      
                      cloudMask = land_cloudy[:,:,-1:]
                      cloudMask[cloudMask!=0]=1
                      
                      land_cloudy2 = self.normalize11(self.scale_landsat_images(np.array(land_cloudy)))
                      
                      #### Binarization Cloud Masks
                      
                      #land_cloudy2[:,:,-1:]=cloudMask
                      landsat_batch_x_cloudy.append(land_cloudy2[:,:,0:8])  # dimension ..*128*128*7  added cloud mask as extra layer
                      
                        # dimension .. 128*128*8
                      tmp_land2 = self.normalize11(self.scale_landsat_images(np.array(tmp_land)))
                      tmp_land2[:,:,-1:] = cloudMask
                      landsat_batch_y_cloud_free.append(tmp_land2)
                      #sentI = self.closest_sentinel(self.get_geohash_from_path(p), timestamp)
                      #landPrev = self.closest_landsat(self.get_geohash_from_path(p),timestamp)
      
                     # if sentI is None or landPrev is None:
                      #    landsat_batch_x_cloudy, landsat_batch_y_cloud_free, g_batch_x,sent_batch,landsat_prev_batch  = [],[],[],[],[]
                      #    break
                      # if landPrev is None:
                      #      break
                      # print("Closest sen: ", sentI.shape)
                      #sent_batch.append(sentI)
                      #landsat_prev_batch.append(landPrev)
                      g_batch_x.append(g_x[index])
      
                      if batch_size is None:
                          batch_size = self.batch_size
      
                      if batch_size == len(landsat_batch_x_cloudy):
                          yield np.array(landsat_batch_x_cloudy), np.array(landsat_batch_y_cloud_free), np.array(g_batch_x)
                          #,np.array(sent_batch),np.array(landsat_prev_batch)
                          landsat_batch_x_cloudy, landsat_batch_y_cloud_free, g_batch_x  = [],[],[]
      

if __name__ == '__main__':
    dataH = DatasetHandling(128, 128, data_type='test',startT='2015-01-01', endT='2020-01-01',
                         album='iowa-2015-2020', no_of_timesteps=3, batch_size=5,node_rank =0)

    it = dataH.load_landsat_images()  # Model 1

    countI = 0
    while True:
         try:
             cloudy_img_land, target_cloudfree_land, cloud_cov_prev = it.__next__()
             countI += len(target_cloudfree_land)
             print("count i = ", countI)
             #print("\nlength of cloudy_img_land:",len(cloudy_img_land))
             #print("\nlength of Target_Cloud_Free:",len(target_cloudfree_land))
             if countI % 5 == 0:
                print("\n Shape of images",cloudy_img_land.shape,target_cloudfree_land.shape)
         except StopIteration:
             break
    print("Total images found: {} ".format(countI))
   #print("Total images found: {} ".format(countI))
