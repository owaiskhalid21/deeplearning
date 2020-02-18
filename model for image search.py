import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle


fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img2/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


img = Image.open('static/img2/sampleblue2.jpg')  # PIL image
uploaded_img_path = "static/uploaded/image.jpg"
img.save(uploaded_img_path, 'JPEG')

query = fe.extract(img)
dists = np.linalg.norm(features - query, axis=1)  # Do search
ids = np.argsort(dists)[:30]  # Top 30 results
scores = [(dists[id], img_paths[id]) for id in ids]

for score in scores:
    print(score)
