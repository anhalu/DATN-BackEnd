import cv2

from reader.base_reader import *


class PaddleTableRec(BaseTable):
    def __init__(self, model_table_name="ppstructure_table", version_table=1, model_rec_name="paddleocr_v3",
                 version_rec=2):
        from paddleocr import PPStructure

        model_table_path = f'models/{model_table_name}_v{version_table}'
        super().__init__(model_table_path=model_table_path)

        model_rec_path = f'models/{model_rec_name}_v{version_rec}'
        self.rec_model_path = os.path.join(self.dir_path, model_rec_path)
        self.rec_char_dict_path = os.path.join(self.rec_model_path, 'word_dict.txt')
        self.model = PPStructure(use_pdserving=False,
                                 use_gpu=True,
                                 lang='en',
                                 rec_model_dir=self.rec_model_path,
                                 rec_char_dict_path=self.rec_char_dict_path,
                                 table_model_dir=self.model_table_path,
                                 show_log=False,
                                 layout=False,
                                 merge_no_span_structure=True)

    def __call__(self, image, *args, **kwargs):
        res = self.model(image, return_ocr_result_in_table=False)[0]['res']['html']
        res = res[
              :6] + '<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Table with Border</title><style>table {border-collapse: collapse;width: 100%;}th, td {border: 1px solid black;padding: 8px;text-align: left;}</style></head>' + res[
                                                                                                                                                                                                                                                                                              6:]
        return res


def test_table():
    m = PaddleTableRec()
    img = cv2.imread(
        '/home/anhalu/anhalu-data/ocr_general_core/data/image/requests/07fad3b6-9bc6-4ef1-8776-f6bcbc4adefa_0.jpg')
    res = m(img)
    print(res)


if __name__ == '__main__':
    test_table()
