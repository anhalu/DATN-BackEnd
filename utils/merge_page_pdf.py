import glob
import os
import tempfile
from typing import Tuple

import cv2
import fitz
import numpy as np
from PIL import Image
from PyPDF2 import PdfFileWriter, PageObject
from PyPDF2 import PdfReader


def mergePage1():
    reader = PdfReader("/Users/tienthien/Downloads/zalo/PT_KS_10 - chua ghep.PDF")

    # write finished pdf
    with open('output.pdf', 'wb') as out_file:
        write_pdf = PdfFileWriter()

        for idx in range(0, reader.getNumPages(), 2):
            page1 = reader.getPage(idx)
            page2 = reader.getPage(idx + 1)

            page3 = PageObject.create_blank_page(None, width=1156 * 2, height=818)
            # create a blankPage size is 17*11,1 inch equal 72 px

            # page3 = page1
            # page3.mergeScaledTranslatedPage(page1, scale=1, tx=0, ty=0, expand=False)
            page3.mergeScaledTranslatedPage(page2, scale=1, tx=1156 * 2, ty=818, expand=False)
            # offset_x = page1.mediaBox[2]
            # print(offset_x)
            # page3.mergeRotatedScaledTranslatedPage(page2, rotation=180, scale=1, tx=1156 * 2, ty=818, expand=True)
            page3.mergeRotatedScaledTranslatedPage(page1, rotation=180, scale=1, tx=0, ty=0, expand=True)
            # merge your pdf1 and pdf2 into the blank canvas
            write_pdf.addPage(page3)
        write_pdf.write(out_file)


def pdf_page_to_image(pdf_path, output_path, dpi=300, extension='.jpg', max_page=-1) -> Tuple[int, str]:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.basename(pdf_path)
    file_id, ext = os.path.splitext(filename)
    if os.path.exists(os.path.join(output_path, f'{file_id}_0{extension}')):
        return 0, file_id
    doc = fitz.open(pdf_path)
    count_page = 0
    for i, page in enumerate(doc):
        # page = doc.load_page(0)  # number of page
        pix = page.get_pixmap(dpi=dpi)
        output_file_path = os.path.join(output_path, f'{file_id}_{i}{extension}')
        pix.save(output_file_path)
        print(f"Save to: {output_file_path}")
        count_page += 1
        if 0 <= max_page <= count_page:
            break
    doc.close()
    return count_page, file_id


def get_table(img, debug=False, sub_img_area=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    if debug:
        cv2.imshow('threshold', threshold)
    # edges = cv2.Canny(gray, 100, 200)
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(threshold, rho=1, theta=np.pi / 180, threshold=50, minLineLength=300, maxLineGap=50)
    if lines is not None:

        if sub_img_area:
            tmp = []
            xmin, ymin, xmax, ymax = sub_img_area
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 20:
                    continue
                if x2 < xmin or x1 > xmax:
                    continue
                if y2 < ymin or y1 > ymax:
                    continue
                tmp.append(line)
            lines = tmp

    if debug:
        # Draw the detected lines on the original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow("line", img)
            cv2.waitKey(0)
        # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # largest_contour = contours[0]
        # x1, y1, w1, h1 = cv2.boundingRect(largest_contour)
        # cv2.rectangle(img, [x1, y1, x1 + w1, y1 + h1], (0, 0, 255), thickness=10)
        # cv2.drawContours(img, [largest_contour], 0, (255, 0, 0), 5)

    # Find the horizontal line closest to the top edge of the image
    top_horizontal_line = None
    top_horizontal_distance = float('inf')
    bottom_horizontal_line = None
    bottom_horizontal_distance = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line is approximately horizontal and near the top edge
            if y1 < top_horizontal_distance:
                top_horizontal_line = line[0]
                top_horizontal_distance = y1
            if y1 > bottom_horizontal_distance:
                bottom_horizontal_line = line[0]
                bottom_horizontal_distance = y1

    x_left = 0
    x_right = img.shape[1]
    # Find the intersection points of the horizontal line with the left and right edges of the image
    y_top_left = 0
    y_top_right = 0
    if top_horizontal_line is not None:
        x1, y1, x2, y2 = top_horizontal_line
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            y_left = int(b)
            y_right = int(m * x_right + b)
        else:
            y_left = y_right = int(y1)
        y_top_left = y_left
        y_top_right = y_right

    y_bottom_left = 0
    y_bottom_right = 0
    if bottom_horizontal_line is not None:
        x1, y1, x2, y2 = bottom_horizontal_line
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            y_left = int(b)
            y_right = int(m * x_right + b)
        else:
            y_left = y_right = int(y1)
        y_bottom_left = y_left
        y_bottom_right = y_right

    if debug:
        cv2.line(img, (x_left, y_top_left), (x_right, y_top_right), (0, 255, 255), 10)
        cv2.line(img, (x_left, y_bottom_left), (x_right, y_bottom_right), (0, 255, 255), 10)

        cv2.imshow("img", img)
        cv2.waitKey(0)
    return y_top_left, y_top_right, y_bottom_left, y_bottom_right


def merge_image(file_path_1, file_path_2, debug=False):
    img1 = cv2.imread(file_path_1)
    img2 = cv2.imread(file_path_2)
    h1, w1 = img1.shape[:2]
    delta1 = int(h1 * 0.02)
    h2, w2 = img2.shape[:2]
    delta2 = int(h2 * 0.02)
    area_size_w = w1 // 3
    y_top_left_1, y_top_right_1, y_bottom_left_1, y_bottom_right_1 = get_table(img1,
                                                                               sub_img_area=[w1 - area_size_w, delta1, w1,
                                                                                             h1-delta1],
                                                                               debug=debug)
    y_top_left_2, y_top_right_2, y_bottom_left_2, y_bottom_right_2 = get_table(img2,
                                                                               sub_img_area=[0, delta2, area_size_w, h2-delta2],
                                                                               debug=debug)

    height_1 = y_bottom_right_1 - y_top_right_1
    height_2 = y_bottom_left_2 - y_top_left_2
    delta_height = abs(height_1 - height_2)
    if height_1 < height_2:
        # resize image 1
        new_height = h1 + delta_height
        new_width = int(w1 * (new_height / h1))
        img1 = cv2.resize(img1, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        h1, w1 = img1.shape[:2]
    elif height_2 > height_1:
        # resize image 2
        new_height = height_2 + delta_height
        new_width = int(w2 * (new_height / h2))
        img2 = cv2.resize(img2, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        h2, w2 = img2.shape[:2]

    delta_y = abs(y_top_right_1 - y_top_left_2)
    new_img_h = max(h1, h2) + delta_y
    new_img_w = w1 + w2
    offset_y_img_1 = 0
    offset_y_img_2 = 0
    if y_top_right_1 < y_top_left_2:
        offset_y_img_1 = delta_y
    else:
        offset_y_img_2 = delta_y

    new_img = np.zeros((new_img_h, new_img_w, 3), dtype=np.uint8)
    new_img[offset_y_img_1:offset_y_img_1 + h1, 0:w1] = img1
    new_img[offset_y_img_2:offset_y_img_2 + h2, w1:] = img2
    new_img = new_img[delta_y:h1 - delta_y, :, :]
    if debug:
        cv2.imshow("test", new_img)
        cv2.waitKey(0)
    return new_img


def merge_a3(input_file_path, output_path: str, from_page: int = 1, max_page=100):
    if os.path.isfile(input_file_path):
        with tempfile.TemporaryDirectory() as temp_dirname:
            print(f"Create temporary directory: {temp_dirname}")

            num_page, file_id = pdf_page_to_image(input_file_path, temp_dirname, dpi=200, extension='.jpg', max_page=-1)
            for i in range(from_page, num_page, 2):
                image = merge_image(os.path.join(temp_dirname, f'{file_id}_{i}.jpg'),
                                    os.path.join(temp_dirname, f'{file_id}_{i + 1}.jpg'))
                cv2.imwrite(os.path.join(output_path, f'{file_id}_{i}.jpg'), image)
    else:
        files = list(glob.glob(os.path.join(input_file_path, f'*.jpg')))
        dir_name = os.path.dirname(files[0])
        base_name = os.path.basename(files[0])
        file_id = os.path.splitext(base_name)[0]
        file_id = file_id[:file_id.rindex('_')]

        idx = from_page
        for _ in range(from_page, len(files), 2):
            file1 = os.path.join(dir_name, f'{file_id}_{idx}.jpg')
            if not os.path.exists(file1):
                idx += 1
                continue
            file2 = os.path.join(dir_name, f'{file_id}_{idx+1}.jpg')
            print(f"Merge page {idx} and page {idx + 1}")
            filename = f'{file_id}_{idx}.jpg'

            image = merge_image(file1, file2)
            cv2.imwrite(os.path.join(output_path, filename), image)
            idx += 2


def image_to_pdf(image_paths, output_file: str):
    print("image to pdf from ")
    images = [Image.open(f) for f in image_paths]
    images[0].save(output_file, save_all=True, append_images=images[1:])


if __name__ == '__main__':
    input_file_path = "/Users/tienthien/workspace/tc_group/data/aToanf/DL_Test_PM/DL_TEST Ghep/Truoc 1999/KB_KS_02_1995.PDF"
    output_path = "KB_KS_02_1995"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    num_page, file_id = pdf_page_to_image(input_file_path, output_path + "tmp", dpi=200, extension='.jpg', max_page=-1)

    merge_a3(output_path + "tmp", output_path, from_page=2)
    image_to_pdf(sorted(glob.glob(f'{output_path}/*.jpg')), f'{output_path}.pdf')
    # for idx in range(46, 80, 2):
    #     print(idx, idx + 1)
    #     merge_image(f'KB_KS_02_1995tmp/KB_KS_02_1995_{idx}.jpg',
    #                 f'KB_KS_02_1995tmp/KB_KS_02_1995_{idx + 1}.jpg', debug=True)
