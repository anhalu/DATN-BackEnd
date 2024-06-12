import glob
import json
import os

import cv2
import fitz


def is_blank_image(image_path, threshold=200, min_black_pixels=1000):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Count the number of black pixels (below the threshold)
    num_black_pixels = cv2.countNonZero(binary_image)

    # Determine if the image is blank based on the count of black pixels
    return num_black_pixels < min_black_pixels


if not os.path.exists('count_dict2.json'):

    dir_path = "/Volumes/Transcend/dữ liệu HP sở xây dựng/HP-7-3-2024/*.pdf"
    result = {}
    for file_path in glob.glob(dir_path):

        count_a3 = 0
        count_a4 = 0
        count_blank = 0

        doc = fitz.open(file_path)
        for page in doc:
            width, height = page.rect.width, page.rect.height
            pix = page.get_pixmap()
            pix.save("page-%i.png" % page.number)
            if is_blank_image("page-%i.png" % page.number):
                print(f"blank page {page.number} file: {file_path}")
                count_blank += 1
                continue
            if width < 800:
                count_a4 += 1
            else:
                count_a3 += 1

        result[file_path] = {
            'A3': count_a3,
            'A4': count_a4,
            'blank': count_blank,
        }
        # print(result)

    json.dump(result, open("count_dict2.json", 'w', encoding='utf8'), ensure_ascii=False)
else:
    result = json.load(open("count_dict2.json", encoding='utf8'))
total_a3 = 0
total_a4 = 0
total_blank = 0
for k, v in result.items():
    total_a3 += v['A3']
    total_a4 += v['A4']
    total_blank += v['blank']
    if v['blank'] > 0:
        print(k, v['blank'])
print(total_a3, total_a4, total_blank)
