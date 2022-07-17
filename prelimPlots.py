import numpy as np
import gdal
import matplotlib.pyplot as plt
import stippy
import socket
from time import strptime, mktime
from skimage import exposure

import datetime
from skimage.transform import resize
import seaborn as sns
import matplotlib.ticker as ticker
import random
import pandas as pd
from scipy.stats.stats import pearsonr
coar = [ '#44AA99', '#DDCC77','#88CCEE', '#117733', '#999933',  '#CC6677', '#882255', '#AA4499']
random.shuffle(coar)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coar)
host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'


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

def getNDSImap(green, swir):
        return (green - swir) / (green + swir)




def getNeigbhours():
        closest = []
        machineId = socket.gethostname().split("-")[-1]
        if machineId == '178':
            closest = ['lattice-178','lattice-179', 'lattice-189','lattice-186']
        elif machineId == '179':
            closest = ['lattice-179', 'lattice-180','lattice-189', 'lattice-186', 'lattice-182']
        elif machineId == '180':
            closest = ['lattice-187', 'lattice-179', 'lattice-180', 'lattice-184', 'lattice-186', 'lattice-182']
        elif machineId == '184':
            closest = ['lattice-180', 'lattice-184', 'lattice-185', 'lattice-182', 'lattice-187', 'lattice-199']
        elif machineId == '185':
            closest = ['lattice-215', 'lattice-184', 'lattice-185', 'lattice-188', 'lattice-187', 'lattice-199']
        elif machineId == '188':
            closest = ['lattice-185', 'lattice-188', 'lattice-215', 'lattice-199']
        elif machineId == '189':
            closest = ['lattice-178','lattice-179', 'lattice-186','lattice-194', 'lattice-193', 'lattice-189']
        elif machineId == '186':
            closest = ['lattice-179', 'lattice-186','lattice-180','lattice-182', 'lattice-194', 'lattice-195']
        elif machineId == '182':
            closest = ['lattice-179', 'lattice-180','lattice-184','lattice-186', 'lattice-182', 'lattice-187',  'lattice-194', 'lattice-195','lattice-201']
        elif machineId == '187':
            closest = ['lattice-185', 'lattice-180', 'lattice-184', 'lattice-199', 'lattice-182', 'lattice-187',
                       'lattice-204', 'lattice-195', 'lattice-201']
        elif machineId == '199':
            closest = ['lattice-184', 'lattice-185', 'lattice-188', 'lattice-187', 'lattice-199', 'lattice-215',
                       'lattice-201', 'lattice-204', 'lattice-205']
        elif machineId == '215':
            closest = ['lattice-185', 'lattice-188', 'lattice-199', 'lattice-215','lattice-204', 'lattice-205']
        elif machineId == '193':
            closest = ['lattice-186', 'lattice-193', 'lattice-194', 'lattice-192']
        elif machineId == '194':
            closest = ['lattice-186', 'lattice-182', 'lattice-193', 'lattice-194','lattice-195', 'lattice-192' , 'lattice-213']
        elif machineId == '195':
            closest = ['lattice-186', 'lattice-182', 'lattice-187', 'lattice-194', 'lattice-195', 'lattice-201',
                       'lattice-192', 'lattice-213', 'lattice-191']
        elif machineId == '201':
            closest = ['lattice-182', 'lattice-187','lattice-199',  'lattice-195', 'lattice-201','lattice-204',
                       'lattice-192', 'lattice-213', 'lattice-191']
        elif machineId == '204':
            closest = ['lattice-187','lattice-199', 'lattice-215',  'lattice-201','lattice-204','lattice-205',
                        'lattice-191','lattice-190', 'lattice-177']
        elif machineId == '205':
            closest = ['lattice-199', 'lattice-215', 'lattice-204', 'lattice-205','lattice-190', 'lattice-177']
        elif machineId == '192':
            closest = ['lattice-193', 'lattice-194', 'lattice-195', 'lattice-192','lattice-213', 'lattice-208', 'lattice-209']
        elif machineId == '213':
            closest = ['lattice-194', 'lattice-195', 'lattice-201','lattice-192','lattice-213', 'lattice-191', 'lattice-208', 'lattice-209', 'lattice-211']
        elif machineId == '191':
            closest = ['lattice-195', 'lattice-201', 'lattice-204','lattice-213', 'lattice-191','lattice-190',
                      'lattice-209', 'lattice-211',  'lattice-212']
        elif machineId == '190':
            closest = ['lattice-201', 'lattice-204','lattice-205','lattice-191', 'lattice-190','lattice-177',
                      'lattice-211', 'lattice-212',  'lattice-214']
        elif machineId == '177':
            closest = ['lattice-204','lattice-205', 'lattice-190','lattice-177', 'lattice-212',  'lattice-214']
        elif machineId == '208':
            closest = ['lattice-192', 'lattice-213','lattice-208','lattice-209', 'lattice-183','lattice-210']
        elif machineId == '209':
            closest = ['lattice-192', 'lattice-213','lattice-191', 'lattice-208','lattice-209', 'lattice-211','lattice-183', 'lattice-210', 'lattice-197']
        elif machineId == '211':
            closest = ['lattice-213', 'lattice-191', 'lattice-190', 'lattice-209', 'lattice-211', 'lattice-212',
                       'lattice-210', 'lattice-197', 'lattice-197', 'lattice-198']
        elif machineId == '212':
            closest = [ 'lattice-191', 'lattice-190','lattice-177',  'lattice-211', 'lattice-212','lattice-214',
                        'lattice-197', 'lattice-197', 'lattice-198','lattice-218']
        elif machineId == '214':
            closest = ['lattice-190', 'lattice-177', 'lattice-212', 'lattice-214', 'lattice-198', 'lattice-218']
        elif machineId == '183':
            closest = ['lattice-208', 'lattice-209', 'lattice-183', 'lattice-210', 'lattice-217', 'lattice-219']
        elif machineId == '210':
            closest = ['lattice-208', 'lattice-209', 'lattice-211', 'lattice-183', 'lattice-210', 'lattice-197', 'lattice-217',  'lattice-219', 'lattice-176']
        elif machineId == '197':
            closest = ['lattice-209', 'lattice-211','lattice-212',  'lattice-210', 'lattice-197','lattice-198',
                       'lattice-206', 'lattice-219', 'lattice-176']
        elif machineId == '198':
            closest = ['lattice-214', 'lattice-211', 'lattice-212', 'lattice-218', 'lattice-197', 'lattice-198',
                       'lattice-206', 'lattice-200', 'lattice-176']
        elif machineId == '218':
            closest = ['lattice-214', 'lattice-212', 'lattice-218', 'lattice-198','lattice-206', 'lattice-200']
        elif machineId == '217':
            closest = ['lattice-217', 'lattice-183', 'lattice-210', 'lattice-219']
        elif machineId == '219':
            closest = ['lattice-183', 'lattice-210',  'lattice-197', 'lattice-176','lattice-217', 'lattice-219']
        elif machineId == '176':
            closest = ['lattice-210', 'lattice-197', 'lattice-198', 'lattice-219',  'lattice-176', 'lattice-206']
        elif machineId == '206':
            closest = ['lattice-218', 'lattice-197', 'lattice-198', 'lattice-200',  'lattice-176', 'lattice-206']
        elif machineId == '200':
            closest = ['lattice-198', 'lattice-218', 'lattice-206', 'lattice-200']

        with open('/s/chopin/a/grad/paahuni/cl/hosts2', "w+") as f:
            for m in closest:
                f.write(m + ' slots=1\n')
            f.close()




def accessFiles(paths, isSentinel=True):
    loadedImages = []
    loaded_image = gdal.Open(paths)
    image = loaded_image.ReadAsArray()  # (11, h, w)
    image = np.moveaxis(image, 0, -1)  # (h, w, 11)
    snowCover = 0
    factor = 0.0002
    if not isSentinel:
        image = image[:, :, [3, 2, 1, 4, 5, 6, 8, 9]]  # Red, Blue, Green, NIR, SWIR 1, SWIR 2, TIR 10, TIR 11
        image = resize(image, (128, 128), preserve_range=True)  # ( 128, 128)
    else:
        newp = paths.replace("-3.tif", "-0.tif")
        loaded_image2 = gdal.Open(newp)
        nirBand = loaded_image2.ReadAsArray()
        nirBand = np.moveaxis(nirBand, 0, -1)[:, :, 3]
        nirBand = resize(nirBand, (128, 128), preserve_range=True) * factor

        newp = paths.replace("-3.tif", "-1.tif")
        loaded_image2 = gdal.Open(newp)
        swirBand = loaded_image2.ReadAsArray()
        swirBand = np.moveaxis(swirBand, 0, -1)[:, :, [4, 5]]
        swirBand = resize(swirBand, (128, 128), preserve_range=True) * factor

        tmp_image = resize(image, (128, 128), preserve_range=True) / 255.0

        snowCover = getSnowCoverge(getNDSImap(tmp_image[:, :, 1] , swirBand[:,:,0]), tmp_image[:, :, 0])
        image = np.dstack((tmp_image, nirBand, swirBand))    # Red, Blue, Green, NIR, SWIR 1, SWIR 2
    loadedImages.append(image)
    return loadedImages, image, snowCover

def getSentinelTileAtGivenTime(maxSentCloud=0.001):
        startT, endT = getTime('2020-01-01', '2021-01-01')
        metaSentinel = {}
        host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        listImages = stippy.list_node_images(host_addr, album='iowa-2015-2020-spa', geocode=None, recurse=False,
                                             platform='Sentinel-2',  max_cloud_coverage = maxSentCloud, min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        try:
            (node, b_image) = listImages.__next__()
            while True:
                for p in b_image.files:
                    if p.path.endswith('-3.tif'):
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


def getSnowCoverge( ndsiMap, redB):
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

def getAllSentiGeo():
        geohashs = []
        startT, endT = getTime('2020-01-01', '2021-01-01')
        host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        listImages = stippy.list_node_images(host_addr, album='iowa-2015-2020-spa', geocode=None, recurse=False,
                                             platform='Sentinel-2', max_cloud_coverage=0.001,
                                             min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        for (node, image) in listImages:
            if image.geocode not in geohashs:
                geohashs.append(image.geocode)
        return geohashs


def train_and_test_landsat_paths(resize_image=True):

            start_date = '2020-01-01'
            end_date = '2021-09-01'
            host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'

            stip_iter_land = stippy.list_node_images(host_addr, platform='Landsat8C1L1', album='iowa-2015-2020-spa',
                                                             min_pixel_coverage=0.9, source='raw',
                                                             start_timestamp=convertDateToEpoch('2020-01-01'), geocode=None,
                                                             end_timestamp=convertDateToEpoch('2021-01-01'), recurse=False,
                                                             max_cloud_coverage=0.001
                                                             )
            input_imgs, timstamps = {}, {}
            SentiP = getAllSentiGeo()

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

            return input_imgs, timstamps


def closest_sentinel(geohash, timestamp, metaSentinel):
    if geohash in metaSentinel.keys():
        times = metaSentinel.get(geohash)
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
            _, image, snowFrac = accessFiles(path_min, isSentinel=True)
            if snowFrac > 0.3:
                for p in allpath:
                    _, image, snowFrac = accessFiles(p, isSentinel=True)
                    if snowFrac < 0.3:
                        return image
                return None
            return image
        else:
            return None
    else:
        return None

def normalize11(imgs):
        """ Returns normalized images between (-1 to 1) pixel value"""
        return imgs / 255.0

def scale_landsat_images(imgs):
        img_tmp = exposure.rescale_intensity(imgs,
                                             out_range=(1, 65535)).astype(np.int)
        img_tmp = np.sqrt(img_tmp).astype(np.int)
        return img_tmp

def getLansdsatTileAtGivenTime(maxLandCloud=0.4):
        startT, endT = getTime('2015-01-01', '2020-01-01')
        metaLandsat = {}
        host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'

        listImages = stippy.list_node_images(host_addr, album='iowa-2015-2020-spa', geocode=None, recurse=False,
                                             platform='Landsat8C1L1',  max_cloud_coverage = 0.001, min_pixel_coverage=0.9,
                                             start_timestamp=startT, end_timestamp=endT)
        try:
            (node, b_image) = listImages.__next__()
            while True:
                for p in b_image.files:
                    if p.path.endswith('0.tif'):
                        pathf = p.path
                        newK = b_image.geocode + "_" + convertEpochToDate(b_image.timestamp)[:4]
                        if newK in metaLandsat.keys():
                            old_data = metaLandsat.get(newK)
                            old_data.append([b_image.timestamp, pathf])
                            metaLandsat[newK] = old_data
                        else:
                            metaLandsat[newK] = [[b_image.timestamp, pathf]]
                (_, b_image) = listImages.__next__()
        except StopIteration:
            return metaLandsat

def closest_landsat(geohash, timestamp, metaLandsat):
            landsat_images_prev = []
            normal_time = convertEpochToDate(timestamp)
            years = ['2015','2016','2017', '2018', '2019']

            for y in years:
                newtimestamp = convertDateToEpoch(y + normal_time[4:])
                newK = geohash + "_" + y
                if newK in metaLandsat.keys():
                        times = metaLandsat.get(newK)
                        min_diff = np.inf
                        path_min = ''
                        for t in times:
                            timeav = t[0]
                            pathav = t[1]
                            if abs(timeav - newtimestamp) <= 86400 * 7 and abs(timeav - newtimestamp) < min_diff:
                                path_min = pathav
                                min_diff = abs(timeav - newtimestamp)
                        if path_min != '':
                            _,im,_ = accessFiles(path_min, isSentinel=False)
                            im = normalize11(scale_landsat_images(im))
                            landsat_images_prev.append(im)
                if len(landsat_images_prev) == 1:
                    break

            if landsat_images_prev == [] or len(landsat_images_prev) < 1:
                return None


            return np.array(landsat_images_prev)

def load_landsat_images(metaLand, metaSentinel, resize_image=True, batch_size=None):
        inp_data_dic, inp_timstamps = train_and_test_landsat_paths(resize_image=True)

        if inp_data_dic is None:
            return

        landsat_batch_y_cloud_free, sent_batch, land_batch = [],[],[]
        g = []
        for geo in inp_data_dic.keys():

            inp_data = inp_data_dic.get(geo)
            for i in range(0, len(inp_data)):

                p = inp_data[i]
                timestamp = inp_timstamps.get(geo)[i]
                sentI = closest_sentinel(geo, timestamp, metaSentinel)
                landI = closest_landsat(geo, timestamp, metaLand)

                if landI is None or sentI is None:
                    continue

                if geo not in g:
                    g.append(geo)
                else:
                    continue

                sent_batch.append(np.array(sentI))
                if landI.shape[0] == 1:
                    landI = np.squeeze(landI, axis=0)
                land_batch.append(np.array(landI))

                rasterL = gdal.Open(p)
                landsat_image_array = rasterL.ReadAsArray().transpose(1, 2, 0)
                if resize_image is True:
                    landsat_image_array = resize(landsat_image_array, (128,128), preserve_range=True)

                # Red, Green, Blue, NIR, TIR
                tmp_land = (landsat_image_array[:, :, [3, 2, 1, 4, 5, 6, 8, 9]])


                landsat_batch_y_cloud_free.append(normalize11(scale_landsat_images(np.array(tmp_land))))

                    # landsat_batch_x_cloudy.append(self.normalize11(self.scale_landsat_images(np.array(land_cloudy))))
                    # landsat_batch_y_cloud_free.append(self.normalize11(self.scale_landsat_images(np.array(tmp_land))))

                if batch_size is None:
                        batch_size = 1

                if batch_size == len(landsat_batch_y_cloud_free):
                        yield np.array(landsat_batch_y_cloud_free), np.array(sent_batch), np.array(land_batch)
                        landsat_batch_y_cloud_free, sent_batch, land_batch = [], [], []

def plot1():
    geohashs = ['9ze', '9zs', '9zk', '9zt', '9zm', '9zw', '9zq']
    monthly_l, monthly_s = {'January':0,'February':0,'March':0,'April':0,'May':0,'June': 0,'July':0,'August':0,'September':0, 'October':0,
                            'November':0, 'December':0},{'January':0,'February':0,'March':0,'April':0,'May':0,'June': 0,'July':0,'August':0,'September':0, 'October':0,
                            'November':0, 'December':0}

    stip_iter_land = stippy.list_images(host_addr, platform='Landsat8C1L1', album='iowa-2015-2020-spa',
                                             min_pixel_coverage=0.9, source='raw',
                                             start_timestamp=convertDateToEpoch('2020-01-01'), geocode=None,
                                             end_timestamp=convertDateToEpoch('2021-01-01'), recurse=False,
                                             max_cloud_coverage=1.0
                                             )
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    count = 0
    for (node, image) in stip_iter_land:
        if image.geocode[:3] in geohashs:
            count += 1
            date = convertEpochToDate(image.timestamp)
            monC = monthly_l.get(switcher.get(int(date[5:7])))
            monC += 1
            monthly_l[switcher.get(int(date[5:7]))] = monC

    print("Total Landsat 8 cloud < 10% images found: ", count)

    stip_iter_land = stippy.list_images(host_addr, platform='Sentinel-2', album='iowa-2015-2020-spa',
                                        min_pixel_coverage=0.9, source='raw',
                                        start_timestamp=convertDateToEpoch('2020-01-01'), geocode=None,
                                        end_timestamp=convertDateToEpoch('2021-01-01'), recurse=False,
                                        max_cloud_coverage=1.0
                                        )
    count = 0
    for (node, image) in stip_iter_land:
        if image.geocode[:3] in geohashs:
            count += 1
            date = convertEpochToDate(image.timestamp)
            monC = monthly_s.get(switcher.get(int(date[5:7])))
            monC += 1
            monthly_s[switcher.get(int(date[5:7]))] = monC

    print("Total Sentinel-2 cloud < 10% images found: ", count)
    print("Monthly Landsat: ", monthly_l)
    print("Monthly Sentinel: ", monthly_s)

    monthly_l_nocloud = [833, 739, 6944, 12404, 2237, 9909, 9863,15419, 10820, 11579, 8790, 3243 ]
    monthly_l_cloud = [24621,18453,18631, 13149,23298,13911,11892,10161,13932, 13990, 7888, 18431]
    monthly_s_nocloud = [6543, 30076, 14324, 17429, 8287, 15771, 11247,6460, 16354,37492, 35018, 25660]
    monthly_s_cloud = [40899,28455, 42910, 26359, 50941, 22386,9118, 10228,41662,41894,40166,49307]

    # Total Landsat 8 cloud < 10% images found: 92780 and >10%: 188357 and TOTAL : 281137
    # Total Sentinel-2 cloud <10% images found: 224661 and >10%: 404325 and TOTAL: 628986
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6), dpi=200)
    land_bar_list = [plt.bar(monthly_l.keys(), monthly_l_cloud, align='edge', width=0.2,label='Landsat 8 Cloud Coverage > 10%'),
                     plt.bar(monthly_l.keys(), monthly_l_nocloud , align='edge', width=0.2,label='Landsat 8 Cloud Coverage <= 10%')
                   ]

    sent_bar_list = [plt.bar(monthly_l.keys(), monthly_s_cloud, align='edge', width=-0.2,label='Sentinel-2 Cloud Coverage > 10%'),
                     plt.bar(monthly_l.keys(), monthly_s_nocloud, align='edge', width=-0.2, label='Sentinel-2 Cloud Coverage <= 10%')]


    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.xlabel("Month")
    plt.ylabel("Number of images")
    plt.title("Number of cloudy and non-cloudy images in Landsat 8 and Sentinel-2 datasets")
    plt.tight_layout()
    plt.savefig("./plots2/1.png")

def plot2():
    host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
    geohashs = ['9ze', '9zs', '9zk', '9zt', '9zm', '9zw', '9zq']

    land_clouds = []
    stip_iter_land = stippy.list_images(host_addr, platform='Landsat8C1L1', album='iowa-2015-2020-spa',
                                        min_pixel_coverage=0.9, source='raw',
                                        start_timestamp=convertDateToEpoch('2020-01-01'), geocode=None,
                                        end_timestamp=convertDateToEpoch('2021-01-01'), recurse=False,
                                        max_cloud_coverage=1.0
                                        )
    for (_, image) in stip_iter_land:
        # if image.geocode[:3] in geohashs:
            land_clouds.append(image.cloudCoverage * 100)
    # print("Got landsat images")
    kwargs = dict(histtype='stepfilled', density=False, bins=20, ec="k")
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.subplot(1, 1, 1)
    plt.title('Cloud Coverage % Distribution for Landsat 8 input dataset')
    plt.ylabel('Probability density for input samples')
    plt.xlabel('Cloud Coverage Percentage (in %)')
    ylabels = ['{:,.2f}'.format(x) + 'K' for x in ax.get_yticks() / 1000]
    ax.hist(land_clouds, **kwargs, label='Landsat 8')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1000) + 'K'))
    plt.tight_layout()

    plt.savefig("./plots2/2.1.png")
    plt.close()
    sent_clouds = []
    stip_iter_land = stippy.list_images(host_addr, platform='Sentinel-2', album='iowa-2015-2020-spa',
                                        min_pixel_coverage=0.9, source='raw',
                                        start_timestamp=convertDateToEpoch('2020-01-01'), geocode=None,
                                        end_timestamp=convertDateToEpoch('2021-01-01'), recurse=False,
                                        max_cloud_coverage=1.0
                                        )
    for (_, image) in stip_iter_land:
        # if image.geocode[:3] in geohashs:
            sent_clouds.append(image.cloudCoverage * 100)
    print("Got Sentinel images")

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6), dpi=200)
    ax = plt.subplot(1, 1, 1)
    plt.title('Cloud Coverage % Distribution for Sentinel-2 input dataset')
    plt.ylabel('no. of input samples')
    plt.xlabel('Cloud Coverage Percentage (in %)')
    ylabels = ['{:,.2f}'.format(x) + 'K' for x in ax.get_yticks() / 1000]
    ax.hist(sent_clouds, **kwargs, label='Sentinel 2', color='#999933')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1000) + 'K'))
    plt.tight_layout()
    # plt.hist(sent_clouds, **kwargs, label='Sentinel-2')
    # plt.legend(title= 'Satellite', loc='best')
    plt.savefig("./plots2/2.2.png")
    plt.close()

def plot3():
    metaSentinel = getSentinelTileAtGivenTime()
    metaLand = getLansdsatTileAtGivenTime()

    count = 0
    bandD = {0: 'Red', 1: 'Green', 2: 'Blue', 3: 'NIR', 4: 'SWIR-1', 5: 'SWIR-2'}
    for bandC in range(6):
        itr = load_landsat_images(metaLand, metaSentinel, batch_size=10)
        globSen, globLand = [], []
        print("Calculating Band {} correlation".format(bandC))
        for (lan, sen, _) in itr:
            sentinel_pixels_bandae = sen[0, :, :, bandC]
            sentinel_pixels_bandae = sentinel_pixels_bandae.flatten()
            sentinel_pixels_bandae = np.clip(sentinel_pixels_bandae, 0, 1)

            landsat_pixels_bandae = lan[0, :, :, bandC]
            landsat_pixels_bandae = landsat_pixels_bandae.flatten()
            landsat_pixels_bandae = np.clip(landsat_pixels_bandae, 0, 1)

            globSen.extend(sentinel_pixels_bandae)
            globLand.extend(landsat_pixels_bandae)

        globLand = np.array(globLand)
        globSen = np.array(globSen)
        cor, _ = pearsonr(globLand, globSen)
        print("Correlation value: ", cor)

        plt.style.use("ggplot")
        plt.scatter(globLand, globSen, s=[1] * len(globSen))
        plt.xlabel('Landsat 8 pixel value (0-1)')
        actC = bandC + 1
        plt.title("Per pixel Correlation between Sentinel-2 and Landsat 8 \n Band " + str(actC) + " (" + bandD.get(
            bandC) + "), Correlation: " + str(round(cor, 4)), fontsize='12')
        plt.ylabel('Sentinel-2 pixel value (0-1)')
        plt.plot(np.unique(globLand), np.poly1d(np.polyfit(globLand, globSen, 1))(np.unique(globLand)), color='black')
        plt.tight_layout()

        plt.savefig("./plots2/3." + str(actC) + ".png")
        plt.close()
    return


def plot4():
    metaLand = getLansdsatTileAtGivenTime()
    metaSentinel = getSentinelTileAtGivenTime()

    count = 0
    bandD = {0:'Red', 1: 'Green', 2: 'Blue', 3: 'NIR', 4:'SWIR-1', 5:'SWIR-2', 6:'TIR-10', 7:'TIR-11'}
    for bandC in range(8):
        itr = load_landsat_images(metaLand, metaSentinel, batch_size=10)
        globLandPrev, globLand = [], []
        for (lan, _, landprev) in itr:
            land_prev_pixels_bandae = landprev[0, :,:, bandC]
            land_prev_pixels_bandae = land_prev_pixels_bandae.flatten()
            land_prev_pixels_bandae = np.clip(land_prev_pixels_bandae, 0, 1)

            landsat_pixels_bandae = lan[0, :, :, bandC]
            landsat_pixels_bandae = landsat_pixels_bandae.flatten()
            landsat_pixels_bandae = np.clip(landsat_pixels_bandae, 0, 1)

            globLandPrev.extend(land_prev_pixels_bandae)
            globLand.extend(landsat_pixels_bandae)
            count += 1

        globLand = np.array(globLand)
        globLandPrev = np.array(globLandPrev)
        cor,_ = pearsonr(globLand, globLandPrev)
        print("Band {} correlation value: {}".format(bandC, cor))

        plt.figure(dpi=200)
        plt.style.use("ggplot")
        plt.scatter(globLand, globLandPrev, s=[1] * len(globLandPrev), c=['#999933'] * len(globLandPrev))
        plt.xlabel('Landsat 8 pixel value (0-1) from previous years')
        actC = bandC + 1
        plt.title("Per pixel Correlation between Landsat 8 in previous years 2015-2019 and 2020\nBands " + str(actC) + " (" + bandD.get(bandC) + "), Correlation: " + str(round(cor, 4)), fontsize='9')
        plt.ylabel('Landsat 8 pixel value (0-1) from 2020')
        plt.plot(np.unique(globLand), np.poly1d(np.polyfit(globLand, globLandPrev, 1))(np.unique(globLand)), color='black')
        plt.tight_layout()
        plt.legend()

        plt.savefig("./plots2/4." + str(actC) + ".png")
        plt.close()
    return

def plot5():
    metaSentinel = getSentinelTileAtGivenTime()
    metaLand = getLansdsatTileAtGivenTime()
    data = {}

    count = 0
    bandD = {0:'Red', 1: 'Green', 2: 'Blue', 3: 'NIR', 4:'SWIR-1', 5:'SWIR-2', 6: 'TIR 10', 7: 'TIR 11'}
    for bandC in range(8):
        itr = load_landsat_images(metaLand , metaSentinel, batch_size=10)
        globLand = []
        for (lan, _, _) in itr:

            landsat_pixels_bandae = lan[0, :, :, bandC]
            landsat_pixels_bandae = landsat_pixels_bandae.flatten()
            # landsat_pixels_bandae = np.clip(landsat_pixels_bandae, 0, 1)

            globLand.extend(landsat_pixels_bandae)
        globLand = np.array(globLand)
        data[bandD.get(bandC)] = globLand

    df = pd.DataFrame(data, columns=['Red', 'Green' ,'Blue', 'NIR','SWIR-1','SWIR-2', 'TIR 10', 'TIR 11'])
    corrMatrix = df.corr()
    print("Corr: ", corrMatrix)

    sns.heatmap(corrMatrix, annot=True)
    plt.tight_layout()
    plt.title("Correlation between bands in Landsat 8 imagery")
    plt.savefig("./plots2/5.png")
    plt.close()
    print("Done!!!!!!!!")

def plot6():
    metaSentinel = getSentinelTileAtGivenTime()
    metaLand = getLansdsatTileAtGivenTime()
    data = {}

    count = 0
    bandD = {0:'Red', 1: 'Green', 2: 'Blue', 3: 'NIR', 4:'SWIR-1', 5:'SWIR-2', 6: 'TIR 10', 7: 'TIR 11'}
    for bandC in range(8):
        itr = load_landsat_images(metaLand , metaSentinel, batch_size=10)
        globLand = []
        for (lan, _, land) in itr:

            landsat_pixels_bandae = lan[0, :, :, bandC]
            landsat_pixels_bandae = landsat_pixels_bandae.flatten()
            # landsat_pixels_bandae = np.clip(landsat_pixels_bandae, 0, 1)

            globLand.extend(landsat_pixels_bandae)
        globLand = np.array(globLand)
        data[bandD.get(bandC)] = globLand

    df = pd.DataFrame(data, columns=['Red', 'Green' ,'Blue', 'NIR','SWIR-1','SWIR-2', 'TIR 10', 'TIR 11'])
    corrMatrix = df.corr()

    sns.heatmap(corrMatrix, annot=True, cmaps='Greens')
    plt.tight_layout()
    plt.title("Correlation between bands in Landsat 8 2020 vs previous years imagery")
    plt.savefig("./plots2/6.png")
    plt.close()
    print("Done!!!!!!!!")


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[  '#CC6677', '#AA4499','#88CCEE', '#44AA99', '#117733', '#999933','#882255', '#DDCC77'])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []


    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys(), loc='best')
    labels = ['Dfb', 'Dwb', 'Dfa', 'Cfa', 'BSk', 'Dsb', 'Csb', 'Csa', 'Dfc', 'Bwh', 'BWk']
    ax.set_xticks([0.0,  1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ax.set_xticklabels(labels)

def plot7():
    geohashs = ['dr9', 'cbt', '9zm', '9ve', '9wy', 'c2m', '9rb', '9r1','9xc', '9mz','9w9']

    l_no_cloud, s_no_cloud, l_cloud, s_cloud = {'dr9':0,'cbt':0,'9zm':0,'9ve':0,'9wy':0,'c2m': 0,'9rb':0,'9r1':0,'9xc':0, '9mz':0,
                            '9w9':0},{'dr9':0,'cbt':0,'9zm':0,'9ve':0,'9wy':0,'c2m': 0,'9rb':0,'9r1':0,'9xc':0, '9mz':0,
                            '9w9':0}, {'dr9':0,'cbt':0,'9zm':0,'9ve':0,'9wy':0,'c2m': 0,'9rb':0,'9r1':0,'9xc':0, '9mz':0,
                            '9w9':0}, {'dr9':0,'cbt':0,'9zm':0,'9ve':0,'9wy':0,'c2m': 0,'9rb':0,'9r1':0,'9xc':0, '9mz':0,
                            '9w9':0}

    stip_iter_land = stippy.list_images(host_addr, platform='Sentinel-2', album='usa-20km-2020',
                                             min_pixel_coverage=0.9, source='raw',
                                             start_timestamp=convertDateToEpoch('2020-05-01'), geocode=None,
                                             end_timestamp=convertDateToEpoch('2020-11-01'), recurse=False,
                                             max_cloud_coverage=1.0
                                             )
    # switcher = {
    #     'dr9' :'Humid Continental Mild Summer, Wet All Year', 'cbt':'Humid Continental Mild Summer With Dry Winters',
    #     '9zm': 'Humid Continental Hot Summers With Year Around Precipitation', '9ve':'Humid Subtropical Climate' ,
    #     '9wy':'Cold Semi-Arid Climate' , 'c2m':'Humid Continental Climate - Dry Cool Summer' , '9rb':'Warm-Summer Mediterranean Climate' ,
    #     '9r1':'Hot-Summer Mediterranean Climate','9xc': 'Subarctic With Cool Summers And Year Around Rainfall',
    #     '9mz': 'Hot Desert Climate','9w9':'Cold Desert Climate'
    # }

    labels = ['Dfb','Dwb','Dfa','Cfa','BSk','Dsb','Csb','Csa','Dfc','Bwh', 'BWk']
    switcher = {'9zm':'Dfa', 'dr9':'Dfb', '9ve':'Cfa', 'cbt':'Dwb', '9wy':'BSk','9w9':'BWk','c2m':'Dsb', '9mz':'Bwh', '9rb':'Csb', '9r1':'Csa', '9xc':'Dfc'}

    count = 0
    for (node, image) in stip_iter_land:
        if image.geocode[:3] in geohashs:
            count += 1
            if image.cloudCoverage >= 0.8 and image.cloudCoverage <= 1.0:
                climateC = l_cloud.get(image.geocode[:3])
                climateC += 1
                l_cloud[image.geocode[:3]] = climateC

    # print("Total Landsat 8 cloud images found: ", count)
    print(l_cloud)
    return
    stip_iter_sent = stippy.list_images(host_addr, platform='Sentinel-2', album='usa-20km-2020',
                                        min_pixel_coverage=0.9, source='raw',
                                        start_timestamp=convertDateToEpoch('2020-05-01'), geocode=None,
                                        end_timestamp=convertDateToEpoch('2021-11-01'), recurse=False,
                                        max_cloud_coverage=1.0
                                        )
    count = 0
    for (node, image) in stip_iter_sent:
        if image.geocode[:3] in geohashs:
            count += 1
            if image.cloudCoverage >= 0.10:
                climateC = s_cloud.get(image.geocode[:3])
                climateC += 1
                s_cloud[image.geocode[:3]] = climateC
            elif image.cloudCoverage < 0.10:
                climateC = s_no_cloud.get(image.geocode[:3])
                climateC += 1
                s_no_cloud[image.geocode[:3]] = climateC

    print("Total Sentinel-2 cloud images found: ", count)


    # monthly_l_nocloud = [833, 739, 6944, 12404, 2237, 9909, 9863,15419, 10820, 11579, 8790, 3243 ]
    # monthly_l_cloud = [24621,18453,18631, 13149,23298,13911,11892,10161,13932, 13990, 7888, 18431]
    # monthly_s_nocloud = [6543, 30076, 14324, 17429, 8287, 15771, 11247,6460, 16354,37492, 35018, 25660]
    # monthly_s_cloud = [40899,28455, 42910, 26359, 50941, 22386,9118, 10228,41662,41894,40166,49307]

    # Total Landsat 8 cloud < 10% images found: 92780 and >10%: 188357 and TOTAL : 281137
    # Total Sentinel-2 cloud <10% images found: 224661 and >10%: 404325 and TOTAL: 628986
    plt.style.use("ggplot")
    t1, t2, t3, t4 = [], [], [], []
    for _,j in l_cloud.items():
        t1.append(j)
    for _,j2 in l_no_cloud.items():
        t2.append(j2)
    plt.figure(figsize=(10, 6), dpi=200)
    land_bar_list = [plt.bar(labels, t1, align='edge', width=0.2,label='Landsat 8 Cloud Coverage >= 10%'),
                     plt.bar(labels, t2 , align='edge', width=0.2,label='Landsat 8 Cloud Coverage < 10%')
                   ]
    for _,j3 in s_cloud.items():
        t3.append(j3)
    for _,j4 in s_no_cloud.items():
        t4.append(j4)
    sent_bar_list = [plt.bar(labels, t3, align='edge', width=-0.2,label='Sentinel-2 Cloud Coverage >= 10%'),
                     plt.bar(labels, t4, align='edge', width=-0.2, label='Sentinel-2 Cloud Coverage < 10%')]

    # plt.yaxis.set_major_formatter(PercentFormatter(xmax=))


    data = {
            "a": [1, 2, 3, 2, 1],
            "b": [2, 3, 4, 3, 1],
            "c": [3, 2, 1, 4, 2],
            "d": [5, 9, 2, 1, 8],
            "e": [1, 3, 2, 2, 3],
            "f": [4, 3, 1, 1, 4],
        }

    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.xlabel("Climate Type")
    plt.ylabel("Number of images")
    plt.title("Number of cloudy and non-cloudy images in Landsat 8 and Sentinel-2 datasets")
    plt.tight_layout()
    plt.savefig("./plots2/7.png")

def plot71():
        data = {
            "0 - 20%": [8885,8728,10153,1939,9619,4284, 0,10403, 0,10137,24431],
            "20 - 40%": [1571,1321,1369,322,287,691, 0,581, 0, 72,691],
            "40 - 60%": [1262,1231,930,184,188,695, 0,640,0,59,544],
            "60 - 80%": [1211,1211,848,171,184,826,0,716,0,51,586],
            "80 - 100%": [9418,11226,14699,867,2787,7443,0,3089,0,730,2233],
        }


        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        bar_plot(ax, data, total_width=.8, single_width=.9, legend=True)

        plt.xlabel("Climate Type")
        plt.ylabel("Number of images")
        plt.title("Number of images with different cloud coverage in \nSentinel-2 over with various climate conditions")
        plt.tight_layout()
        plt.savefig("./plots2/7.2.png")
from tensorflow.keras.models import Sequential, model_from_json,Model


def getLayerWeightChange(model1, model2):
    path = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"
    json_file = open(path + str(model1) + "/GeneratorModel.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model1 = model_from_json(loaded_model_json)
    loaded_model1.load_weights(path + str(model1)+ "/GeneratorModel.h5")
    print(loaded_model1.summary())
    json_file = open(path + str(model2) + "/GeneratorModel.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model2 = model_from_json(loaded_model_json)
    loaded_model2.load_weights(path + str(model2) + "/GeneratorModel.h5")

    weightChange, layerNames = [],[]
    for layer1, layer2 in zip(loaded_model1.layers, loaded_model2.layers):
        we1 = layer1.get_weights()
        we2 = layer2.get_weights()
        if len(we1) != 0:
            weight1 = np.array(we1[0])
            biases1 = np.array(we1[1])

            weight2 = np.array(we2[0])
            biases2 = np.array(we2[1])

            if np.isnan(weight1).all() or np.isnan(weight2).all():
                continue

            err = np.mean(np.abs(weight1 - weight2))

            if np.isnan(err).all():
                continue
            weightChange.append(err)
            layerNames.append(layer1.name)
            # print(layer1.name, " Mean absoulute error : ", err)
    zipped_lists = zip( weightChange, layerNames)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [element for _, element in sorted_zipped_lists]
    sorted_list2 = [element for element, _ in sorted_zipped_lists]
    print(sorted_list1)
    # print("N: ", sorted_list2)

if __name__ == '__main__':
            print("Select your plot number")
            # plot7()

            # Landsat 8
            # data = {
            #     "0 - 20%": [6706, 5790, 8903, 1909, 11339,5691,6256,15725,1501,13977,14211],
            #     "20 - 40%": [1264, 774,  845,  529,   242, 247, 418,  686, 291,  158,  522],
            #     "40 - 60%": [1248, 752,  958,  340,   204, 237, 412,  545, 293,  194,  392],
            #     "60 - 80%": [1161, 855,  958,  267,   234, 362, 525,  517, 337,  199,  430],
            #     "80 - 100%":[7583, 11521,8417, 1202, 1364, 9109,6652, 2263,3072, 763,  2797],
            # }
            getLayerWeightChange(1000, 99)



