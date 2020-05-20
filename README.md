# Plant pathology classification

## Context
This is a little project to practice images classification and state-of-the-art CNN architecture implementation on a small dataset. The data is from the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) hosted on Kaggle.

The dataset is composed of two sets of 1841 high definition images of apple tree leafs, covering four categories of them: healthy, scab infected, rust infected and infected with multiple diseases. The goal is map unseen images to those classes.

## Project structure
The repository is organized around a notebook implementing each steps of the exercise:
* Loading data, and instantiating the custom class representing training, validation and test datasets, implemented in **dataset.py**. Pytorch is the main library used for modeling, and as such a derivate of its abstract class Dataset is used to define images transformations and supply (sample, label) tuples for training and validation, or sample for testing. the package **albumentations** is used to transform the images. Because the training set is rather small, a simple oversampling strategy is implemented: each image is used several times with randomized transforms. Some images and their respective transformed versions are shown.

* Modeling, training and evaluation. The model is a custom implementation of the wide variant of the ResNet architecture, as described in the paper [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v2.pdf) and found in **models.py**. This project being mainly to practice pytorch and CNN architecture, no pre-trained or pre-implemented models are used. These architecture having a lot of parameters and thus tends to easily overfit, only a rather shallow one is used here as the dataset is quite small.

For readability, most of the code is packaged dedicated files:
* dataset custom class and data utilities in dataset.py
* Wide ResNet implementation in models.py

## Getting Started

Just clone this repository:
```
git clone https://github.com/clabrugere/plant-pathology-classification.git
```

Download the dataset from Kaggle and put it in:
```
data/
```

### Requirements

* Python 3.7
* numpy
* pandas
* plotly
* matplotlib
* pytorch
* sklearn

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)
* [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v2.pdf)
