import os
from typing import List, Tuple, Any

import numpy as np
from loguru import logger

from layout_parser.elements import Document


class Vocab(object):
    def __init__(self, vocab_path='models/word_dict.txt', *args, **kwargs):
        self.character_str = ['blank']
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if vocab_path is None:
            raise FileNotFoundError("vocab path not found : {}".format(vocab_path))
        if not vocab_path.startswith('/'):
            vocab_path = os.path.join(self.dir_path, vocab_path)
        self.vocab_path = vocab_path

        logger.info(f"Initialize vocab with vocab path = {vocab_path}")

        if self.vocab_path is not None:
            with open(self.vocab_path, 'rb') as f:
                lines: List[bytes] = f.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip('\r\n')
                    self.character_str.append(line)
            self.character_str.append(' ')
            self.len_char = len(self.character_str) - 1
            self.idx2ch = {idx: self.character_str[idx] for idx in range(self.len_char + 1)}
            self.ch2idx = {self.character_str[idx]: idx for idx in range(self.len_char + 1)}

    def __len__(self):
        return len(self.character_str)

    def encode(self, word):
        return [self.ch2idx[ch] for ch in word]

    def encode_batch(self, batch):
        return [self.encode(word) for word in batch]

    def decode(self, idxs: List):
        return [self.idx2ch[idx] for idx in idxs]

    def decode_batch(self, batch):
        return [self.decode(idxs) for idxs in batch]

    def decode_ctc(self, texts):
        res = ''
        for idx_text, text in enumerate(texts):
            if idx_text < len(texts) - 1 and text == texts[idx_text + 1]:
                continue
            if text == 'blank':
                continue
            res += text
        return res

    def decode_ctc_batch(self, batch):
        return [self.decode_ctc(texts) for texts in batch]


class BaseOnnx(object):
    def __init__(self, model_path: str = None, providers: list = None, *args, **kwargs):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not model_path.startswith('/'):
            model_path = os.path.join(self.dir_path, model_path)
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Not found model onnx file at {self.model_path}")

        if not self.model_path.endswith('.onnx'):
            raise TypeError("Only onnx models are supported")

        logger.info(f"Initializing onnx model with model path = {self.model_path}")
        self.providers = providers

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def rec_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def resize_norm_img(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class BaseDet(object):
    def __init__(self, model_path: str, score_threshold=0.01, img_size=1024, iou=0.5,
                 max_det=3000):
        self.score_threshold = score_threshold
        self.img_size = img_size
        self.iou = iou
        self.max_det = max_det
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not model_path.startswith('/'):
            model_path = os.path.join(self.dir_path, model_path)
        self.model_path = model_path
        self.model = None

        logger.info(f'Initialize model Detect with model path = {model_path}')

    def __str__(self):
        return self.model_path

    def __call__(self, images: List[np.ndarray], return_image=False, *args, **kwargs) -> List[
        List[Tuple[int, str, float, List, Any]]]:
        """
        Predict all image and return list box of prediction
        :param images: List images input in np.ndarray
        :param return_image: crop image of prediction and return
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()


class BaseTable(object):
    def __init__(self, model_table_path, *args, **kwargs):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not model_table_path.startswith('/'):
            model_table_path = os.path.join(self.dir_path, model_table_path)
        self.model_table_path = model_table_path
        self.model = None

        logger.info(f'Initialize model Table Recognite with model path = {model_table_path}')

    def __str__(self):
        return self.model_table_path

    def __call__(self, image, *args, **kwargs):
        raise NotImplementedError


class BaseRec(object):
    def __init__(self, model_path, score_threshold=0.2):
        self.score_threshold = score_threshold
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not model_path.startswith('/'):
            model_path = os.path.join(self.dir_path, model_path)
        self.model_path = model_path
        self.model = None

        logger.info(f'Initialize model Recognite with model path = {model_path}')

    def __str__(self):
        return self.model_path

    def __call__(self, images: List[np.ndarray], *args, **kwargs):
        raise NotImplementedError()


class BaseClassifier(object):
    def __init__(self, model_path, score_threshold=0.2, *args, **kwargs):
        self.score_threshold = score_threshold
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not model_path.startswith('/'):
            model_path = os.path.join(self.dir_path, model_path)
        self.model_path = model_path
        self.model = None
        logger.info(f'Initialize model Classification with model path = {model_path}')

    def __str__(self):
        return self.model_path

    def __call__(self, images: List, *args, **kwargs):
        raise NotImplementedError()


class BaseReader(object):
    def __init__(self, *args, **kwargs):
        pass

    def init_model(self, *args, **kwargs):
        raise NotImplementedError()

    def _detect_text(self, images: List[np.ndarray], *args, **kwargs):
        raise NotImplementedError()

    def _recognite_text(self, images: List[np.ndarray], *args, **kwargs):
        raise NotImplementedError()

    def read_images(self, images: List[np.ndarray], return_merge=True, *args, **kwargs) -> Document:
        raise NotImplementedError()

    def predict_layout(self, images, *args, **kwargs):
        raise NotImplementedError()

    def read_batches(self, images: List[np.ndarray], debug=False, *args, **kwargs):
        raise NotImplementedError()


class BaseCorrect(object):
    def __init__(self, length_filter=10):
        self.length_filter = length_filter
        self.model = None

    def __call__(self, texts: List[str], *args, **kwargs):
        raise NotImplementedError()
#
#
# class VietnameseCorrector(BaseCorrect):
#     def __init__(self, length_filter=10):
#         super().__init__(length_filter=length_filter)
#
#         from transformers import pipeline
#
#         self.model = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
#
#     def __call__(self, texts: List[str], *args, **kwargs):
#         start_time = time.time()
#         res = []
#         for text in texts:
#             if len(text) > self.length_filter:
#                 text = self.model(text)
#             res.append(text)
#         end_time = time.time()
#         print(f"Correct {len(texts)} texts: return {len(texts)} texts in {end_time - start_time:.5}s")
#
#         return res
#
#
# def test_vietnamese_correct():
#     m = VietnameseCorrector()
#     text = """
#              IV. Những thay đổi sau khi cấp Giấy chứng nhận
#                                    Xác nhận của cơ quan
#        Nội dung thay đổi và cơ sở pháp lý
# Người được cấp Giấy chứng nhận không được sửa chữa, tấy xóa hoặc bồ
# sung bất kỳ nội dung nào trong Giấy chứng nhận; khi bị mất hoặc hư
# hỏng Giấy chứng nhận phải khai báo ngay với cơ quan cấp Giấy.
#     """
#     texts = text.split('\n')
#     print(m(['CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM']))
