import time

import numpy as np
from paddleocr.tools.infer.predict_rec import TextRecognizer


class CustTextRecognizer(TextRecognizer):
    def __init__(self, args):
        super(CustTextRecognizer, self).__init__(args)

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        pred_probs = []
        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = []
                gsrm_word_pos_list = []
                gsrm_slf_attn_bias1_list = []
                gsrm_slf_attn_bias2_list = []
            if self.rec_algorithm == "SAR":
                valid_ratios = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "SAR":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "SRN":
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25)
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
                elif self.rec_algorithm in ["SVTR", "SATRN"]:
                    norm_img = self.resize_norm_img_svtr(img_list[indices[ino]],
                                                         self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm in ["VisionLAN", "PREN"]:
                    norm_img = self.resize_norm_img_vl(img_list[indices[ino]],
                                                       self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == 'SPIN':
                    norm_img = self.resize_norm_img_spin(img_list[indices[ino]])
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "ABINet":
                    norm_img = self.resize_norm_img_abinet(
                        img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "RobustScanner":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]],
                        self.rec_image_shape,
                        width_downsample_ratio=0.25)
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios = []
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                    word_positions_list = []
                    word_positions = np.array(range(0, 40)).astype('int64')
                    word_positions = np.expand_dims(word_positions, axis=0)
                    word_positions_list.append(word_positions)
                elif self.rec_algorithm == "CAN":
                    norm_img = self.norm_img_can(img_list[indices[ino]],
                                                 max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                    norm_image_mask = np.ones(norm_img.shape, dtype='float32')
                    word_label = np.ones([1, 36], dtype='int64')
                    norm_img_mask_batch = []
                    word_label_list = []
                    norm_img_mask_batch.append(norm_image_mask)
                    word_label_list.append(word_label)
                else:
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = {"predict": outputs[2]}
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = {"predict": outputs[2]}
            elif self.rec_algorithm == "SAR":
                valid_ratios = np.concatenate(valid_ratios)
                inputs = [
                    norm_img_batch,
                    np.array(
                        [valid_ratios], dtype=np.float32),
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "RobustScanner":
                valid_ratios = np.concatenate(valid_ratios)
                word_positions_list = np.concatenate(word_positions_list)
                inputs = [norm_img_batch, valid_ratios, word_positions_list]

                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "CAN":
                norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
                word_label_list = np.concatenate(word_label_list)
                inputs = [norm_img_batch, norm_img_mask_batch, word_label_list]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs
                else:
                    input_names = self.predictor.get_input_names()
                    input_tensor = []
                    for i in range(len(input_names)):
                        input_tensor_i = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor_i.copy_from_cpu(inputs[i])
                        input_tensor.append(input_tensor_i)
                    self.input_tensor = input_tensor
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs
            else:
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    image_shape = [(x.shape[2], x.shape[1]) for x in norm_img_batch]
                    # print(image_shape)
                    if image_shape[0][0] < 1000:
                        self.input_tensor.copy_from_cpu(norm_img_batch)
                        self.predictor.run()
                        # get image padding shape
                        # print(image_shape)
                        outputs = []
                        for output_tensor in self.output_tensors:
                            output = output_tensor.copy_to_cpu()
                            outputs.append(output)
                        if self.benchmark:
                            self.autolog.times.stamp()
                        if len(outputs) != 1:
                            preds = outputs
                        else:
                            preds = outputs[0]
                        self.predictor.try_shrink_memory()
                        # print(preds.shape, image_shape, indices[range(beg_img_no, end_img_no)])
                        # print(type(preds))
                        # print(type(image_shape), len(image_shape))
                    else:
                        # ---------------------------------------------------------------------------------------------------------------------------
                        # fix bug tam thoi
                        # ---------------------------------------------------------------------------------------------------------------------------
                        # print("????")
                        preds = np.zeros((len(image_shape), 1, 230), dtype=np.float32)
                    # print(image_shape)
                    pred_probs.append((preds, image_shape, indices[range(beg_img_no, end_img_no)]))
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if self.benchmark:
                self.autolog.times.end(stamp=True)

        return pred_probs, rec_res, time.time() - st
