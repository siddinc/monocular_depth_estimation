# Monocular Depth Estimation
The goal of this project is to implement the paper on "High Quality Monocular Depth Estimation via Transfer Learning" by Ibraheem Alhashim and Peter Wonka and replace their DenseNet architecture by a UNet with a ResNet encoder.

>  I have trained a Unet Convolutional Neural Network with a Resnet encoder (pre-trained on imagenet weights) on the NYU-Depth v2 dataset which obtained a soft accuracy of 83% on the test set.
> The output is demonstrated below where the left image represents the predicted depth map, the middle image represents the ground truth depth map and the right image is the ground truth scene. The color map shown in the output is "Plasma" from Matplotlib color maps. Brighter the color, the nearer the object and darker the color, the farther away is the object.

<img src="./images/output.png" width="300" height="300" />

## Tech used:
- TensorFlow 2.0.0
- Python 3.5.6

## Instructions to run:
- Using `anaconda`:
  - Run `conda create --name <env_name> --file recog.yml`
  - Run `conda activate <env_name>`
- Using `pip`:
  - Run `pip install -r requirements.txt`
- `cd` to `src`
- Run `python main.py

## Reference:
@article{Alhashim2018,
  author    = Ibraheem Alhashim and Peter Wonka,
  title     = High Quality Monocular Depth Estimation via Transfer Learning,
  journal   = arXiv e-print,
  volume    = abs/1812.11941,
  year      = 2018,
  url       = https://arxiv.org/abs/1812.11941,
  eid       = arXiv:1812.11941,
  eprint    = 1812.11941
}