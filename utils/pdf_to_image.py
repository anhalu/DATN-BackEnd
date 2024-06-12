# STEP 1
# import libraries
import glob
import os.path

import fitz
from pdf2image import convert_from_path


def get_image_from_pdf2(pdf_path, output_path):
    filename = os.path.basename(pdf_path)
    file_id, ext = os.path.splitext(filename)
    output_file_path = os.path.join(output_path, f'{file_id}.jpg')

    pages = convert_from_path(pdf_path, dpi=600)

    for count, page in enumerate(pages):
        page.save(f'{output_file_path}', 'JPEG')


def pdf_page_to_image(pdf_path, output_path, dpi=300):
    filename = os.path.basename(pdf_path)
    file_id, ext = os.path.splitext(filename)
    if os.path.exists(os.path.join(output_path, f'{file_id}_0.jpg')):
        return
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        # page = doc.load_page(0)  # number of page
        pix = page.get_pixmap(dpi=dpi)
        output_file_path = os.path.join(output_path, f'{file_id}_{i}.jpg')
        pix.save(output_file_path)
        print(f"Save to: {output_file_path}")
    doc.close()


if __name__ == '__main__':
    for file in glob.glob("/Volumes/Transcend/DL_SO HO TICH/**/*.pdf", recursive=True):
        image_path = os.path.splitext(file)[0] + '.jpg'
        print(file)
        output_dir = '/Volumes/Transcend/DL_SO HO TICH/images/'
        pdf_page_to_image(file, output_dir)
