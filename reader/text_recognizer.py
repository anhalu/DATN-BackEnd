import math
import time
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
import paddle
import torch
from PIL import Image
from loguru import logger

from reader.base_reader import BaseRec, BaseOnnx


class PaddleRec(BaseRec):
    def __init__(self, model_name='paddleocr_v3', version=1, batch_size=6, score_threshold=0.2):
        from paddleocr.paddleocr import parse_args
        from paddleocr.tools.infer.predict_rec import TextRecognizer

        model_path = f'models/{model_name}_v{version}'
        super().__init__(model_path=model_path, score_threshold=score_threshold)

        params = parse_args(mMain=False)
        params.rec_model_dir = self.model_path
        params.rec_char_dict_path = f'{self.model_path}/word_dict.txt'
        params.rec_batch_num = batch_size
        params.rec_image_shape = "3,32,320"
        params.lang = 'vi'
        params.use_gpu = True
        params.enable_mkldnn = True
        params.warmup = True
        self.model = TextRecognizer(params)

    def __call__(self, images: List[np.ndarray], *args, **kwargs) -> Tuple[List, List]:
        start_time = time.time()
        results, exe_time = self.model(images)
        texts = []
        probs = []
        for result in results:
            texts.append(result[0])
            probs.append(result[1])
        end_time = time.time()
        logger.info(f"Rec {len(images)} images: return {len(probs)} objects in {end_time - start_time:.5}s")
        return texts, probs, None


class CustomPaddleRec(BaseRec):
    def __init__(self, model_name='paddleocr_v3', version=1, batch_size=6, score_threshold=0.2):
        from paddleocr.paddleocr import parse_args
        from reader.custom_text_recognizer_paddle_ocr import CustTextRecognizer

        model_path = f'models/{model_name}_v{version}'
        super().__init__(model_path=model_path, score_threshold=score_threshold)

        params = parse_args(mMain=False)
        params.rec_model_dir = self.model_path
        params.rec_char_dict_path = f'{self.model_path}/word_dict.txt'
        params.rec_batch_num = batch_size
        params.rec_image_shape = "3,32,320"
        params.lang = 'vi'
        params.enable_mkldnn = True
        params.warmup = True
        self.model = CustTextRecognizer(params)

    def decode_batch(self, preds):
        """
            Decodes a batch of predictions into a list of words.

            Args:
                preds (List[np.ndarray]): A list of predictions, each represented as a
                    numpy array of integers.

            Returns:
                List[List[str]]: A list of words, where each word is represented as a
                    list of characters.
            """
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=-1)
        # preds_prob = preds.max(axis=2)

        return [[self.model.postprocess_op.character[x] for x in pred] for pred in preds_idx]

    def __call__(self, images: List[np.ndarray], *args, **kwargs) -> Tuple[List, List, List]:
        start_time = time.time()

        preds, results, exe_time = self.model(images)
        char_batches = [[]] * len(images)
        image_shapes = [(0, 0)] * len(images)
        for pred_batch, image_shape_batch, index in preds:
            batch_chars = self.decode_batch(pred_batch)
            for chars, image_shape, ind in zip(batch_chars, image_shape_batch, index):
                char_batches[ind] = chars
                image_shapes[ind] = image_shape

        coord_splits_all = []
        for chars, image_shape, result, image in zip(char_batches, image_shapes, results, images):
            num_chars = len(chars)
            img_w, img_h = image_shape
            orig_img_h, orig_img_w = image.shape[:2]
            step_width = img_w / num_chars

            tmp_spaces = []
            start_space_index = -1
            start_char = 0
            end_char = len(chars)
            for i, c in enumerate(chars[::-1]):
                if c != 'blank' and c != ' ':
                    end_char = len(chars) - i
                    break
            end_char = max(end_char + 3, len(chars))
            for i, c in enumerate(chars):
                if c == ' ' and start_space_index < 0:
                    start_space_index = i
                    continue
                if c != 'blank' and c != ' ' and start_space_index >= 0:
                    tmp_spaces.append((start_space_index, i))
                    start_space_index = -1
            coord_splits = [0]
            for start_space_index, end_space_index in tmp_spaces:
                coord_split = (end_space_index + start_space_index) / 2 * step_width
                coord_split = coord_split / img_h * orig_img_h
                coord_splits.append(coord_split)
            coord_splits.append(orig_img_w)
            coord_splits_all.append(coord_splits)

        # for image, coord_splits in zip(images, coord_splits_all):
        #     img_h, img_w = image.shape[:2]
        #     for coord_split in coord_splits:
        #         x = int(coord_split)
        #         cv2.line(image, (x, 0), (x, img_h), (255, 0, 0), thickness=2)
        #     cv2.imshow('test', image)
        #     cv2.waitKey(0)

        texts = []
        probs = []
        for result in results:
            texts.append(result[0])
            probs.append(result[1])
        end_time = time.time()

        logger.info(f"Rec {len(images)} images: return {len(probs)} objects in {end_time - start_time:.5}s")
        return texts, probs, coord_splits_all


class VietOcrRec(BaseRec):
    def __init__(self, model_name='vgg_seq2seq', version=1, algo='vgg_seq2seq', score_threshold=0.0):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.translate import process_input

        model_path = f'models/{model_name}_v{version}.pth'
        super().__init__(model_path=model_path, score_threshold=score_threshold)

        self.rec_config = Cfg.load_config_from_name(algo)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rec_config['device'] = device
        self.rec_config['cnn']['pretrained'] = False
        self.model = Predictor(self.rec_config)
        self.process_input = process_input()

    def __call__(self, images: List[np.ndarray], *args, **kwargs) -> Tuple[List, List]:
        start_time = time.time()
        images = [Image.fromarray(i) for i in images]

        texts, probs = self.model.predict_batch(images, return_prob=True)
        end_time = time.time()
        print(f"Rec {len(images)} images: return {len(probs)} objects in {end_time - start_time:.5}s")

        return texts, probs

    def predict_batch(self, images: List[np.ndarray], *args, **kwargs) -> Tuple[List, List]:
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}
        for i, img in enumerate(images):
            img = self.process_input(img, self.rec_config['dataset']['image_height'],
                                     self.rec_config['dataset']['image_min_width'],
                                     self.rec_config['dataset']['image_max_width'])

            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)


class OCROnnx(BaseOnnx):
    def __init__(self, model_name='paddleocrv4_onnx', version=1, providers: list = ['CPUExecutionProvider'],
                 rec_shape=(3, 48, 5, 1072)):
        import onnxruntime

        model_path = f'models/{model_name}_v{version}.onnx'
        super().__init__(model_path=model_path, providers=providers)
        self.rec_shape = rec_shape
        self.rec_model = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        self.rec_input_names = [x.name for x in self.rec_model.get_inputs()]
        self.rec_output_names = [x.name for x in self.rec_model.get_outputs()]

    def predict(self, batch):
        norm_img_batch = self.rec_preprocess(batch)
        rec_input_dict = {'x': norm_img_batch}
        output = self.rec_model.run(self.rec_output_names, rec_input_dict)[0]
        pred_idxs = output.argmax(axis=2)
        pred_probs = output.max(axis=2)
        return pred_idxs, pred_probs

    def rec_preprocess(self, batch, in_model=True):
        imgC, imgH, imgW_min, imgW_max = self.rec_shape
        batch_n = []
        for img in batch:
            h = img.shape[0]
            w = img.shape[1]
            ratio = imgH / float(h)
            resize_w = math.ceil(w * ratio)
            resize_h = imgH
            resized_image = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            batch_n.append(resized_image)

        max_w_batch = None
        for img in batch_n:
            w = img.shape[1]
            if max_w_batch is None:
                max_w_batch = w
            else:
                max_w_batch = max(max_w_batch, w)
        res_batch = []
        for resized_image in batch_n:
            resize_h = resized_image.shape[0]
            resize_w = resized_image.shape[1]
            if in_model:  # infer co norm
                resized_image = resized_image.astype('float32')
                resized_image = resized_image.transpose((2, 0, 1)) / 127.5
                resized_image -= 1  # -1 -> 1
                padding_img = np.ones((imgC, imgH, max_w_batch), dtype=np.float32)
                padding_img[:, :, 0:resize_w] = resized_image
                res_batch.append(padding_img)
            else:  # ko norm
                padding_img = 255 * np.ones((imgH, max_w_batch, imgC))
                padding_img[:, :resize_w, :] = resized_image
                res_batch.append(padding_img)
        return res_batch

    def post_process(self, img, vocab):
        [img] = self.rec_preprocess([img], in_model=False)
        results = self.predict([img])
        vocab = vocab
        texts = vocab.decode(results[0][0])
        ctc_texts = vocab.decode_ctc(texts)
        # print(texts)
        # print(ctc_texts)
        ctc_words = ctc_texts.split()
        ctc_len = len(ctc_words)
        # probs = results[1][0]
        distance = len(texts)
        # imgH = img.shape[0]
        imgW = img.shape[1]
        distance_step = imgW / distance
        list_coor = [0]
        i = 1
        word_count = 0
        while i < distance:
            if texts[i] == ' ':
                tmp = []
                for j in range(i, distance):
                    if texts[j] == ' ':
                        tmp.append(j * distance_step + distance_step / 2)
                    else:
                        i = j - 1
                        break
                if len(tmp) > 0:
                    list_coor.append(np.mean(tmp))
                    word_count += 1
            if word_count >= ctc_len:
                break
            if i == distance - 1 and texts[i] != ' ':
                list_coor.append(distance * distance_step / imgW)
                word_count += 1

            i += 1

        # for idx_coor, coord in enumerate(list_coor):
        #     if idx_coor < len(list_coor) - 1:
        #         cv2.rectangle(img, (int(list_coor[idx_coor]), 0), (int(list_coor[idx_coor + 1]), imgH),
        #                       generate_random_color(), thickness=2)

        # cv2.imwrite('test_step1.jpg', img)

        return ctc_words, list_coor


def generate_random_color():
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def test():
    model = OCROnnx(providers=["CUDAExecutionProvider"])
    img = cv2.imread('/home/anhalu/anhalu-data/github/word_reg/longword/51dbe8ab-67c2-4a54-bc9b-9d3675b35cc9/0.jpg')
    model.post_process(img)

    exit(0)
    [img] = model.rec_preprocess([img], in_model=False)
    results = model([img])
    vocab = Vocab()

    texts = vocab.decode(results[0][0])
    ctc_texts = vocab.decode_ctc(texts)
    print(texts)
    print(ctc_texts)

    probs = results[1][0]
    distance = len(texts)
    print(len(texts))
    imgH = img.shape[0]
    imgW = img.shape[1]
    distance_step = imgW / distance
    list_coor = [0]
    i = 1
    while i < distance:
        if texts[i] == ' ':
            tmp = []
            for j in range(i, distance):
                if texts[j] == ' ':
                    tmp.append(j * distance_step + distance_step / 2)
                else:
                    i = j - 1
                    break
            if len(tmp) > 0:
                list_coor.append(np.mean(tmp))
        if i == distance - 1 and texts[i] != ' ':
            list_coor.append(distance * distance_step)

        i += 1
    print(len(list_coor), len(ctc_texts.split(' ')))
    print(list_coor)
    for idx_coor, coord in enumerate(list_coor):
        if idx_coor < len(list_coor) - 1:
            cv2.rectangle(img, (int(list_coor[idx_coor]), 0), (int(list_coor[idx_coor + 1]), imgH),
                          generate_random_color(), thickness=2)

    cv2.imwrite('test_step1.jpg', img)


def test_vietocr_rec(imgs=None):
    m = VietOcrRec(model_name='vgg_seq2seq', version=1, algo='vgg_seq2seq', score_threshold=0)
    print(m)
    if imgs:
        texts, probs = m(imgs)
        for text, prob in zip(texts, probs):
            print(f'\t\t{prob:.2} - {text}')
            break


def test_paddle_rec(imgs=None):
    m = PaddleRec(batch_size=6)
    print(m)
    if imgs:
        texts, probs = m(imgs)
        for text, prob in zip(texts, probs):
            print(f'\t\t{prob:.2} - {text}')


def test_cus_paddle_rec(imgs=None):
    m = CustomPaddleRec(version=2, batch_size=6)
    print(m)
    if imgs:
        texts, probs = m(imgs)
        for text, prob in zip(texts, probs):
            print(f'\t\t{prob:.2} - {text}')
            break


if __name__ == '__main__':
    imgs = [
        cv2.imread(f'/home/anhalu/anhalu-data/github/word_reg/longword/51dbe8ab-67c2-4a54-bc9b-9d3675b35cc9/{i}.jpg')
        for i in range(10)]

    test_cus_paddle_rec(imgs)
