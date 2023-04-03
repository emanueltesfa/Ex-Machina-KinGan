# Ex-Machina - Highly Variable Kin Face Image Synthesis
- [Presentation Slides](https://docs.google.com/presentation/d/1te6pDOuLYa86QUPClomKFSSpHMiRBk8AmDqGvF_ou7o/edit?usp=sharing)
- [Report](https://www.overleaf.com/project/642475fcb89d84398363ba6c)
## Timeline and Goals

* By midterm (date?):
  1. Replicate results from KinshipGEN and GANKIN
  2. Goal of Project
  3. Experiment with different Neural Network architectures for kin feature predictors
* Bonus
  1. Improve upon GAN used to generate faces

## Description

# The image-synthesis pipeline is composed of a couple models.

Primarly, we are using **StyleDNA**.
Here is how you can build your own pipeline.

## Get Started
### StyleDNA
#### Prerequisites

- Linux or macOS
- Python3
- PyTorch == 1.9.0+cu111
- dlib == 19.22.1

#### Pretrained Model Weights

- [Face Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

  Shape predictor from dlib.

- [StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

  StyleGAN2 model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.

- [Other pretrained model weights](https://drive.google.com/drive/folders/1ExZtCMFeLP4y5VYNg9rQWnkBCxbQ38xc?usp=sharing)

#### Inference Notebook

We provide a Jupyter notebook version running on [Google Colab](https://colab.research.google.com/drive/1FHf5ftbYtAfvODEqj5lp-S1cir44UniT?usp=sharing) for fast inferecing.

## Inferencing

#### Pretrained Weights

Download all the pretrained model weights and put them in *./pretrained_model/*

#### Inferencing

Having your trained model weight, you can use `./inference.py` to test the model on a set of images.
For example,
```
python3 inference.py --mom_path ./test/mom.png --dad_path ./test/dad.png
```

## Credits
 - stylegan2: https://github.com/rosinality/stylegan2-pytorch  
 - pSp: https://github.com/eladrich/pixel2style2pixel  
 - InsightFace_Pytorch: https://github.com/TreB1eN/InsightFace_Pytorch  
 - dlib: http://dlib.net/face_landmark_detection.py.html  
 - Face alignment: https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5

