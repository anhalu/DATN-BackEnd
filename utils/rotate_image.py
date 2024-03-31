import glob

from PIL import Image

for file in glob.glob('/Users/tienthien/Downloads/data_scan/scan111/*.jpg', recursive=True):
    print("Rotating image: " + file)
    try:
        img = Image.open(file)
        # rotate and save the image with the same filename
        img.rotate(-90, expand=True).save(file)
        # close the image
        img.close()
    except:
        pass
