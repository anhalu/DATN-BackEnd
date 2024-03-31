import json
label_files = ['/content/ocr_general/generate/vi_doc/output', '/content/ocr_general/generate/vi_doc_v2/output']
train_rate = 0.8
f_label_train = open("/content/ocr_general/generate/output/label_train.txt", 'w')
f_label_test = open("/content/ocr_general/generate/output/label_test.txt", 'w')

for label_file in label_files:
    labels = json.load(open(f'{label_file}/labels.json', 'r', encoding='utf8'))
    images_folder = f'{label_file}/images/'
    num_example = len(labels['labels'])
    num_train = int(num_example * train_rate)
    idx = 0
    for file_id, label in labels['labels'].items():
        if idx < num_train:
            f_label_train.write(f'{images_folder}{file_id}.jpg\t{label}\n')
        else:
            f_label_test.write(f'{images_folder}{file_id}.jpg\t{label}\n')
        idx += 1
f_label_train.close()
f_label_test.close()