class DatasetHandling():

    def __init__(self, w=128, h=128, no_of_timesteps=2,
                 pix_cover=1.0, startT='2015-01-01', endT='2020-01-01',
                 cloud_cov=0.3, album='iowa-2015-2020', batch_size=1, istrain=True):

        self.targetH = h
        self.targetW = w
        self.startT = startT
        self.endT = endT
        self.pix_cover = pix_cover
        self.cloud_cov = cloud_cov
        self.no_of_timesteps = no_of_timesteps
        self.album = album
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        self.metaSentinel = self.getSentinelTileAtGivenTime()
        self.ccThreshTest = 0.9
        self.ccThreshTrain = 0.2
        self.batch_size = batch_size
        self.lenarr=[]
        self.istrain = istrain

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


    def getSentinelTileAtGivenTime(self, maxSentCloud=0.3):
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
            return metaSentinel

    def getAllSentiGeo(self):
        geohashs = []
        startT, endT = self.getTime('2020-01-01', '2021-01-01')
        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=None, recurse=False,
                                             platform='Sentinel-2', max_cloud_coverage=0.15,
                                             min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        for (node, image) in listImages:
            if image.geocode not in geohashs:
                geohashs.append(image.geocode)

        return geohashs


    def accessFiles(self, paths, isSentinel=True):
            loadedImages = []
            loaded_image = gdal.Open(paths)
            image = loaded_image.ReadAsArray() #(11, h, w)
            image = np.moveaxis(image, 0, -1)  #(h, w, 11)

            factor = 2 / 1e4
            if not isSentinel:
                image = image[:,:,[3,2,1,4,5,6,8,9]]  #Red, Blue, Green, NIR, SWIR 1, SWIR 2, TIR 10, TIR 11
                image = resize(image, (self.targetW, self.targetH), preserve_range=True)  # ( 128, 128)
            else:
                newp = paths.replace("-0.tif", "-1.tif")
                loaded_image2 = gdal.Open(newp)
                swirBand = loaded_image2.ReadAsArray()
                swirBand = np.moveaxis(swirBand, 0, -1)[:,:, [4, 5]]
                swirBand = resize(swirBand, (self.targetW, self.targetH), preserve_range=True)
                tmp_image = image[:, :, [0,1,2,3]]
                tmp_image = resize(tmp_image, (self.targetW, self.targetH), preserve_range=True)
                image = np.dstack((tmp_image, swirBand)) * factor
            loadedImages.append(image)
            return loadedImages,image

    def closest_sentinel(self, geohash, timestamp):

        if geohash in self.metaSentinel.keys():
            times = self.metaSentinel.get(geohash)

            min_diff = np.inf
            path_min = ''
            for t in times:
                timeav = t[0]
                pathav = t[1]

                if abs(timeav - timestamp) <= 86400 * 6 and abs(timeav - timestamp) < min_diff:
                    path_min = pathav
                    min_diff = abs(timeav - timestamp)

            if path_min != '':
                _, image = self.accessFiles(path_min, isSentinel=True)
                return image
            else:
                return None
        else:
            return None

    def train_and_test_landsat_paths(self, resize_image=True):
        '''
        This method returns paths to cloud mask and training target images paths
        :param ccThreshTest: maximum cloud coverage for training data (remove clouds for image this ccThreshTrain + 0.5 <= cc <= ccThreshTest)
        :param ccThreshTrain: maximum cloud coverage for testing data (target for images used for training cc <= ccThreshTrain)
        :return: [[path, cloud_coverage],...] array of paths to cloud mask, [path1, ...] array of paths to clean target images
        '''

        if self.istrain:
            start_date = '2020-01-01'
            end_date = '2021-09-01'
        else:
            start_date = '2020-09-02'
            end_date = '2021-01-01'

        cloud_masks = {}
        stip_iter_land = stippy.list_node_images(self.host_addr, platform='Landsat8C1L1', album=self.album,
                                                 min_pixel_coverage=self.pix_cover, source='raw',
                                                 start_timestamp=self.convertDateToEpoch(start_date), geocode=None,
                                                 end_timestamp=self.convertDateToEpoch(end_date), recurse=False,
                                                 max_cloud_coverage=0.60
                                                 )
        # from 0.15 to 1.0 cloud coverage
        for (node, image) in stip_iter_land:
            # rn = random.random()
            if image.cloudCoverage >= 0.30 and image.cloudCoverage < 0.60:
                if image.files[0].path.endswith("-0.tif"):
                    p = image.files[0].path
                    if image.geocode in cloud_masks.keys():
                        paths = cloud_masks.get(image.geocode)
                        if len(paths > 10):
                            continue
                        paths.append([p, image.cloudCoverage])
                        cloud_masks[image.geocode] = paths
                    else:
                        cloud_masks[image.geocode] = [[p, image.cloudCoverage]]
                else:
                    continue

        if len(cloud_masks) == 0 or cloud_masks == {}:
            return None, None, None, None, None

        stip_iter_land = stippy.list_node_images(self.host_addr, platform='Landsat8C1L1', album=self.album,
                                                 min_pixel_coverage=self.pix_cover, source='raw',
                                                 start_timestamp=self.convertDateToEpoch('2020-01-01'), geocode=None,
                                                 end_timestamp=self.convertDateToEpoch('2021-01-01'), recurse=False,
                                                 max_cloud_coverage=0.05
                                                 )
        input_imgs, timstamps = {}, {}
        SentiP = self.getAllSentiGeo()

        for (node, image) in stip_iter_land:

            if image.geocode not in SentiP:
                continue

            if image.files[0].path.endswith("-0.tif"):
                p = image.files[0].path
                if image.geocode in input_imgs.keys():
                    paths = input_imgs.get(image.geocode)
                    paths.append(p)
                    input_imgs[image.geocode] = paths

                    timstampsA = timstamps.get(image.geocode)
                    timstampsA.append(image.timestamp)
                    timstamps[image.geocode] = timstampsA
                else:
                    input_imgs[image.geocode] = [p]
                    timstamps[image.geocode] = [image.timestamp]  # Number of clean image for given geohash
            else:
                continue

        x_cloud_dic, g_x_dic = {}, {}

        for g in cloud_masks:
            x_cloud, g_x = [], []
            for (pathCM, cc) in cloud_masks.get(g):
                raster = gdal.Open(pathCM)
                actualCloudPixel = raster.ReadAsArray().transpose(2, 1, 0)

                if resize_image is True:
                    actualCloudPixel = resize(actualCloudPixel, (self.targetW, self.targetH), preserve_range=True)

                cloud_mask_array = actualCloudPixel[:, :, 10]
                cm = np.empty((self.targetW, self.targetH, 8))

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        # set ice/snow pixels as 0
                        if (2800 <= cloud_mask_array[i][j] < 3744) or (7840 > cloud_mask_array[i][j] > 3788) or \
                                cloud_mask_array[i][j] >= 7872:
                            # or (cloud_mask_array[i][j] == 6896) or (cloud_mask_array[i][j] == 6900) or (cloud_mask_array[i][j] == 6904) or (cloud_mask_array[i][j] == 6908) :
                            cm[i, j] = actualCloudPixel[i, j, [3, 2, 1, 4, 5, 6, 8, 9]]
                        else:
                            cm[i, j] = [0, 0, 0, 0, 0, 0, 0, 0]

                x_cloud.append(cm)
                g_x.append(float(cc))

            x_cloud_dic[g] = x_cloud
            g_x_dic[g] = g_x

        return cloud_masks, input_imgs, timstamps, x_cloud_dic, g_x_dic

    def normalize11(self, imgs):
        """ Returns normalized images between (-1 to 1) pixel value"""
        return imgs / 127.5 - 1

    def denormalize11(self, imgs):
        return (imgs + 1.) * 127.5

    def scale_landsat_images(self, imgs):
        img_tmp = exposure.rescale_intensity(imgs,
                                             out_range=(1, 65535)).astype(np.int)
        img_tmp = np.sqrt(img_tmp).astype(np.int)
        return img_tmp
    
    def load_landsat_images(self, resize_image=True, batch_size=None):
        cloud_masks, inp_data_dic, inp_timstamps, x_cloud, g_x = self.train_and_test_landsat_paths(resize_image=True)

        if cloud_masks is None or inp_data_dic is None or x_cloud is None:
            return

        landsat_batch_x_cloudy, landsat_batch_y_cloud_free = [], []

        for geo in inp_data_dic.keys():
            if geo not in x_cloud.keys():
                continue
            inp_data = inp_data_dic.get(geo)
            for i in range(0, len(inp_data)):

                p = inp_data[i]
                timestamp = inp_timstamps.get(geo)[i]
                sentI = self.closest_sentinel(geo, timestamp)

                if sentI is None:
                    continue

                rasterL = gdal.Open(p)
                landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                if resize_image is True:
                    landsat_image_array = resize(landsat_image_array, (self.targetW, self.targetH), preserve_range=True)

                # Red, Green, Blue, NIR, TIR
                tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4, 5, 6, 8, 9]])

                for (index, cloudm) in enumerate(x_cloud.get(geo)):
                    land_cloudy = copy.deepcopy(tmp_land)

                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):

                            if not all(cloudm[i, j] == 0):
                                land_cloudy[i, j] = cloudm[i, j, :]

                    landsat_batch_x_cloudy.append(self.normalize11(self.scale_landsat_images(np.array(land_cloudy))))
                    landsat_batch_y_cloud_free.append(self.normalize11(self.scale_landsat_images(np.array(tmp_land))))

                    if batch_size is None:
                        batch_size = self.batch_size

                    if batch_size == len(landsat_batch_x_cloudy):
                        yield np.array(landsat_batch_x_cloudy), np.array(landsat_batch_y_cloud_free), geo
                        landsat_batch_x_cloudy, landsat_batch_y_cloud_free = [], []

