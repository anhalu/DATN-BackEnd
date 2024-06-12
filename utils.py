import os
import re
import shutil
import time
from functools import wraps
from typing import List

import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass

import fitz
from PIL import Image
import cv2

from loguru import logger


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        if 'debug' in kwargs and kwargs['debug']:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.debug(f'Function {func.__name__} Took {total_time:.4f} seconds')
        else:
            result = func(*args, **kwargs)
        return result

    return timeit_wrapper


def get_image_from_pdf2(pdf_path) -> List:
    imgs = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    return imgs


def get_image_from_pdf3_v2(pdf_path, save_folder=None, dpi=200):
    logger.info(f"PDF path: {pdf_path} to : {save_folder}")
    filename = os.path.basename(pdf_path)
    file_id, ext = os.path.splitext(filename)
    i = 0
    start = time.time()
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            output_name = os.path.join(save_folder, f'{file_id}_{i}.jpg')
            s = time.time()
            cv2.imwrite(output_name, img)
            # logger.info('--------------------- : Time write to file : ', time.time() - s)
            i += 1
    logger.info("+++++++++++++++++++++++++++ : ALL TIME FOR V2 : ", time.time() - start)
    with open(os.path.join(save_folder, f'{file_id}_report.txt'), 'w') as f:
        f.write(f"Total pages: {i}")

    return i
    # imgs.append(img)
    # return imgs


# def get_image_from_pdf3_v3(pdf_path, save_folder=None, dpi=200):
#     logger.info(f"PDF path: {pdf_path} to : {save_folder}")
#     # imgs = [vips_to_numpy(pyvips.Image.new_from_file(filename, dpi=300, page=i)) for i in range(n_pages)]
#     filename = os.path.basename(pdf_path)
#     file_id, ext = os.path.splitext(filename)
#     import time
#     start = time.time()
#     image = pyvips.Image.new_from_file(pdf_path)
#     nums = image.get('n-pages')
#     for i in range(nums):
#         image = pyvips.Image.new_from_file(pdf_path, page=i, dpi=dpi)
#         if save_folder:
#             s = time.time()
#             image.write_to_file(os.path.join(save_folder, f'{file_id}_{i}.jpg'))
#             logger.info('--------------------- : Time write to file : ', time.time() - s)
#
#     logger.info("+++++++++++++++++++++++++++ : ALL TIME FOR PYVIP V3 : ", time.time() - start)
#     with open(os.path.join(save_folder, f'{file_id}_report.txt'), 'w') as f:
#         f.write(f"Total pages: {nums}")
#     logger.info(f"Total pages: {nums}")
#     return nums
#

# def get_image_from_pdf3(pdf_path, save_folder=None, dpi=200):
#     logger.info(f"PDF path: {pdf_path} to : {save_folder}")
#     # imgs = [vips_to_numpy(pyvips.Image.new_from_file(filename, dpi=300, page=i)) for i in range(n_pages)]
#     filename = os.path.basename(pdf_path)
#     file_id, ext = os.path.splitext(filename)
#     i = 0
#     import time
#     start = time.time()
#     while True:
#         try:
#             s = time.time()
#             image = pyvips.Image.new_from_file(pdf_path, dpi=dpi, page=i)
#             logger.info('=========***** : Time pyvips new form file :', time.time() - s)
#             output_name = os.path.join(save_folder, f'{file_id}_{i}.jpg')
#             if save_folder:
#                 s = time.time()
#                 image.write_to_file(output_name)
#                 # image = image.numpy()
#                 # cv2.imwrite(output_name, image)
#                 print('--------------------- : Time write to file : ', time.time() - s)
#             i += 1
#         except Exception as e:
#             # print(e)
#             break
#
#     logger.info("+++++++++++++++++++++++++++ : ALL time white true pyvips : ", time.time() - start)
#     with open(os.path.join(save_folder, f'{file_id}_report.txt'), 'w') as f:
#         f.write(f"Total pages: {i}")
#     logger.info(f"Total pages: {i}")
#     return i


def check_line_in_block(line, block):
    return line.min_x >= block.min_x and line.max_x <= block.max_x and line.min_y >= block.min_y and line.max_y <= block.max_y


def merge_line_to_block(page, debug=False):
    if len(page.blocks) == 0 or len(page.lines) == 0:
        return

    for line in page.lines:
        for block in page.blocks:
            if check_line_in_block(line, block):
                if debug:
                    cv2.rectangle(page.img, (int(line.min_x), int(line.min_y)), (int(line.max_x), int(line.max_y)),
                                  (36, 255, 12), 2)
                    cv2.rectangle(page.img, (block.min_x, block.min_y), (block.max_x, block.max_y), (255, 10, 12), 2)

                    cv2.imshow('image', page.img)
                    cv2.waitKey()
                block.lines.append(line)


def remove_non_block(page):
    remove = []
    for block in page.blocks:
        if len(block.lines) == 0:
            remove.append(block)
    for block in remove:
        page.blocks.remove(block)


def crop_4_points(image, boxes):
    rect = boxes
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(np.array(rect, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped, rect


def remove_lower_prob_lines(page, prob_threshold=0.3):
    remove = []
    for line in page.lines:
        if line.prob < prob_threshold:
            remove.append(line)
    for line in remove:
        page.lines.remove(line)
    return page


def plot_block(page, level=1, line=False):
    img = page.img
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    if line:
        for i, l in enumerate(page.lines):
            cv2.rectangle(img, (int(l.min_x), int(l.min_y)), (int(l.max_x), int(l.max_y)), colors[0], 1)
            img = cv2.putText(img, f'{i}', (int(l.min_x), int(l.min_y)), cv2.FONT_HERSHEY_SIMPLEX,
                              1, colors[0], 1, cv2.LINE_AA)

    for i, block in enumerate(page.blocks):
        cv2.rectangle(img, (int(block.min_x), int(block.min_y)), (int(block.max_x), int(block.max_y)), colors[1], 1)
        img = cv2.putText(img, f'{i}', (int(block.min_x), int(block.min_y)), cv2.FONT_HERSHEY_SIMPLEX,
                          1, colors[1], 1, cv2.LINE_AA)
    plt.imshow(img)
    plt.show()


def insert_image_to_pdf(images: List, filename=None):
    doc = fitz.open()

    for img in images:
        rect = fitz.Rect(0, 0, img.shape[1], img.shape[0])

        page = doc.new_page(width=rect.width, height=rect.height)
        retval, buffer = cv2.imencode(".jpg", img)
        image_bytes = buffer.tobytes()

        page.insert_image(rect, stream=image_bytes)
    doc.save(filename)


def split_block_text(text: str):
    text = text.strip()
    lines = text.split('\n')
    lines_result = []
    consider = []
    for line in lines:
        res = re.split('\s{10,}', line.strip())
        logger.info(res)
        if len(res) > 1:
            if consider:  # nếu trong danh sách đang xét có phần tử
                if len(res) == len(consider[-1]):
                    # cùng số lượng phần tử thì thêm
                    consider.append(res)
                else:
                    # không cùng số lượng phần tử thì tiến hành tách danh sách cũ và tạo danh sách mới
                    for i in range(len(consider[0])):
                        for l in consider:
                            lines_result.append(l[i])
                    consider.clear()
                    consider.append(res)

            else:  # k có bản ghi trong danh sách đang xét
                consider.append(res)
        else:
            if consider:
                for i in range(len(consider[0])):
                    for l in consider:
                        lines_result.append(l[i])
                consider.clear()
            if len(res[0]) > 0:  # nếu chỉ dó line text thì thêm vào lines_result
                lines_result.append(res[0])

    logger.info("\n\n\n================\n\n\n\n")
    for l in lines_result:
        logger.info(l)


def all_same(items):
    return all(x == items[0] for x in items)


class NotSupportFormat(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_images_local(file_path, file_id, file_ext, saved_folder):
    if file_ext in ['pdf']:
        import time
        start = time.time()
        if not os.path.exists(f'{saved_folder}/{file_id}_report.txt'):
            num_pages = get_image_from_pdf3_v2(pdf_path=file_path, save_folder=saved_folder)
        i = 0
        start = time.time()
        while True:
            if os.path.exists(f'{saved_folder}/{file_id}_{i}.jpg'):
                yield cv2.imread(f'{saved_folder}/{file_id}_{i}.jpg')
            else:
                break
            i += 1

    elif file_ext in ['jpg', 'jpeg', 'png']:
        if not os.path.exists(f'{saved_folder}/{file_id}_0.jpg'):
            shutil.copyfile(file_path, f'{saved_folder}/{file_id}_0.jpg')

        yield cv2.imread(file_path)
    else:
        raise NotSupportFormat(f"Not support format {file_ext}")


if __name__ == '__main__':
    split_block_text("""

E
i


       Vi . Những thay đối sau khi cấp Giấy chứng nhận thuy
      Nội dung thay đổi và cơ sở pháp lý                           Xác nhận của cơ chận
                            th. có thẩm
Người được cấp Giấy chứng nhận không được sửa chừa, tấy xóa hoặc bồ
sung bất kỳ nội dung nào trong Giấy chứng nhận; khi bị mất hoặc hư
hóng Giấy chứng nhận phải khai báo ngay với cơ quan cấp Giấy.


      CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
           Độc lập - Tự do - Hạnh phúc
       GIẤY CHỨNG NHẬN
           QUYỀN SỬ DỤNG ĐẤT
QUYỀN SỞ HỮU NHÀ Ở VÀ TÀI SẢN KIỀN VỚI ĐẤT
 1. Người sử dụng đất, chủ sở hữu nhà ở và tài sản khác gắn liền với đất
 Bà Trần Thị Ngát                    Sinh năm: 1956
 CMND số: 121 000 008 cấp ngày 13/11/2009 tại công an tinh Bắc Giang.
 Địa chỉ thường trú: Xã Đông Phú, huyện Lục Nam, tỉnh Bắc Giang
                            BX 420381


D
""")
