import matplotlib.pyplot as plt
import numpy as np
import gdal

if __name__ == '__main__':
    p = gdal.Open("/s/chopin/a/grad/paahuni/NLCD_2019_Land_Cover_L48_20210604_IhzIhnAhwvh4bHv3MecV.tiff")
    im = p.ReadAsArray()
    label, counts = np.unique(im, return_counts=True)
    labels = {11:'Open Water', 21: 'Developed, Open Space', 22:'Developed, Low Intensity', 23:'Developed, Medium Intensity',
              24:'Developed, High Intensity', 31:'Barren Land (Sand/Rock/Clay)', 41:'Deciduous Forest',
              42:'Evergreen Forest', 43:'Mixed Forest', 52:'Shrub/Scrub', 71: 'Grassland/Herbaceous',
              81:'Pasture/Hay', 82:'Cultivated Crops', 90:'Woody Wetlands', 95:'Emergent Herbaceous Wetlands'}
    print(np.sort(counts))

    labelsF = labels.get(11),labels.get(21),labels.get(22),labels.get(23),labels.get(24),labels.get(31),labels.get(41),labels.get(42),labels.get(43),labels.get(52),labels.get(71), labels.get(81),labels.get(82), labels.get(90), labels.get(95)
    print(labelsF)
    sizes = counts
    explode = (0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labelsF, autopct='%1.1f%%',
             shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('./landcoverTypes.png')