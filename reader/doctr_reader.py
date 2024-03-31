from typing import List

import numpy as np
from doctr.io import read_img_as_numpy
from doctr.models import ocr_predictor, db_resnet50_rotation, crnn_vgg16_bn

from layout_parser.elements import Document
from reader.base_reader import BaseReader


class DocTrReader(BaseReader):
    def __init__(self):
        super().__init__()
        self.det_model = db_resnet50_rotation(pretrained=True)
        self.rec_model = crnn_vgg16_bn(pretrained=True)
        self.ocr_model = ocr_predictor(det_arch=self.det_model,
                                       reco_arch=self.rec_model,
                                       pretrained=True,
                                       assume_straight_pages=True,
                                       preserve_aspect_ratio=True,
                                       symmetric_pad=False,
                                       detect_orientation=True,
                                       detect_language=True)

    def read_images(self, images: List[np.ndarray], return_merge=True) -> Document:
        return self.ocr_model(images)


if __name__ == '__main__':
    img_path = '../data/image/requests/2dcb247f-a012-4ba9-98ab-6a2fccf32489_0.jpg'
    image = read_img_as_numpy(img_path)
    model = DocTrReader()
    res = model.read_images([image])
    res.show([image])
