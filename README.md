# Sis: Simple Image Search Engine

## [Demo](http://www.simple-image-search.xyz/)
![](http://yusukematsui.me/project/sis/img/screencapture2.jpg)

## Workflow
![](http://yusukematsui.me/project/sis/img/overview.jpg)

## Overview
- *Sis* is a simple image-based image search engine using Keras + Flask. You can launch the search engine just by running two python scripts.
- `offline.py`: This script extracts deep features from images. Given a set of database images, a 4096D fc6-feature is extracted for each image using the VGG16 network with ImageNet pre-trained weights.
- `server.py`: This script runs a web-server. You can send your query image to the server via a Flask web-intereface. Then relevant images to the query are retrieved by the simple nearest neighbor search.
- On an aws-ec2 instance with t2.large, the feature extraction takes 0.9 s per image. The search for 1000 images takes 10 ms. We tested Sis on Ubuntu 16.04 with Python3.
