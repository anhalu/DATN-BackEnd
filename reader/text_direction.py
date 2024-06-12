import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms

from reader.base_reader import *


class TextDirectionReader(BaseClassifier):
    def __init__(self, model_name='vgg19_bn', version=1, score_threshold=0.7, labels=None):
        model_path = f'models/{model_name}_v{version}.pt'
        super().__init__(model_path, score_threshold)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(self.model_path, map_location=self.device)
        self.img_transforms = transforms.Compose([
            transforms.Resize((60, 60)),
            transforms.ToTensor()
        ])
        if labels is not None:
            self.labels = labels
        else:
            self.labels = {
                0: 0,
                1: -90,
                2: 180,
                3: 90
            }

    def predict(self, images: List, batch):
        batch_images = []
        list_outputs = []
        for idx, image in enumerate(images):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            batch_images.append(self.img_transforms(image))
            if len(batch_images) >= batch:
                batch_image = torch.stack(batch_images)
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(batch_image.to(self.device))
                    outputs = nn.functional.softmax(outputs, dim=1).detach().cpu().numpy().tolist()
                    list_outputs.extend(outputs)
                    batch_images.clear()
        if len(batch_images) > 0:
            batch_image = torch.stack(batch_images)
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(batch_image.to(self.device))
                outputs = nn.functional.softmax(outputs, dim=1).detach().cpu().numpy().tolist()
                list_outputs.extend(outputs)
                batch_images.clear()
        return self.get_labels(list_outputs)

    def __call__(self, images: List, *args, **kwargs):
        return self.predict(images)

    def get_labels(self, outputs):
        idxs_max = np.argmax(outputs, axis=1)
        results_label = []
        results_prob = []
        for i, idx in enumerate(idxs_max):
            results_label.append(self.labels[idx])
            results_prob.append(round(outputs[i][idx], 2))
        return results_label, results_prob

    def rotate(self, images: List, batch=None):
        if batch is None:
            batch = len(images)
        results_label, results_prob = self.predict(images, batch)
        list_images = []
        for label, prob, image in zip(results_label, results_prob, images):
            angle = 0
            logger.info(f"=============== direction image prob : {prob}, label : {label}")
            if prob > self.score_threshold:
                angle = label
            rotated_image = image.rotate(angle, expand=True)
            list_images.append(rotated_image)
        return list_images


def test():
    labels = {
        0: 0,
        1: -90,
        2: 180,
        3: 90
    }
    model = TextDirectionReader(labels=labels)
    img = Image.open('/home/anhalu/anhalu-data/ocr_general_core/di.jpg').convert('RGB')
    # results_label, results_prob = model.predict([img, img])
    # print(results_label)
    list_images = model.rotate([img, img, img], batch=2)
    # for img_in in list_ima
    #             image_soto = []


if __name__ == '__main__':
    test()
