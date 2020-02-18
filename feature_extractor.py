from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
import numpy as np


class FeatureExtractor:
    def __init__(self):
        base_model = InceptionResNetV2(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((299, 299))  # Inception must take a 299x299 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize

