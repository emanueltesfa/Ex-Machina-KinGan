# Ex-Machina - Highly Variable Kin Face Image Synthesis
- [Presentation Slides](https://docs.google.com/presentation/d/1te6pDOuLYa86QUPClomKFSSpHMiRBk8AmDqGvF_ou7o/edit?usp=sharing)
- [Report](https://www.overleaf.com/project/642475fcb89d84398363ba6c)
- [Quick Inference Notebook](https://www.kaggle.com/code/keycasey/kin-faces)

This image synthesis pipeline is composed of a couple models.

Primarly, we are using **StyleDNA**.
Here is how you can build your own pipeline.

## Get Started
### StyleDNA
#### Prerequisites

- Linux or macOS
- Python3
- PyTorch == 1.9.0+cu111
- dlib == 19.22.1

## Ex-Machina Data sources
### Fiw Dataset:
  - https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/data
  - You will need folders "test-public-lists" and 'train-faces'
    store as:

```
recognizing-faces-in-the-wild/
  test-public-lists/
  train-faces/
```
### FFHQ Aging  Dataset: 
  - https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
  - subset of FFHQ dataset (due to disk storage constraints)
  - store as:
```
FFHQ-Aging-Dataset/
  data/
```



## Pretrained Model Weights
### Download models
1. Before running Inference.py or any training scripts you must run the gdown cell in quick_styledna_inference.ipynb to get all pretrained models except model_ir_se50.pth (ArcFace).

2. change the name of 60.pth to 10.pth
3. Download arcface model from [here](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&id=CEC0E1F8F0542A13%21835&cid=CEC0E1F8F0542A13&parId=root&parQt=sharedby&parCid=155373F2BD163C07&o=OneUp)
3. store models as:
```
pretrained_model/
  10.pth
  condi_4_2.pth
  enc_4_2.pth
  model_ir_se50.pth
  shape_predictor_68_face_landmarks.dat
  stylegan2-ffhq-config-f.pt
```
- [Face Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

  Shape predictor from dlib.

- [StyleGAN2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)

  StyleGAN2 model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
  StyleGAN2 model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.

- [Other pretrained model weights](https://drive.google.com/drive/folders/1ExZtCMFeLP4y5VYNg9rQWnkBCxbQ38xc?usp=sharing)

#### Inference Notebook

We provide a Jupyter notebook version running on [Google Colab](https://colab.research.google.com/drive/1FHf5ftbYtAfvODEqj5lp-S1cir44UniT?usp=sharing) for fast inferecing.

## Inferencing

#### Pretrained Weights

Download all the pretrained model weights and put them in *./pretrained_model/*

#### Inferencing

Having your trained model weight, you can use `./inference.py` to test the model on a set of images.
Having your trained model weight, you can use `./inference.py` to test the model on a set of images.
For example,
```
python3 inference.py --mom_path ./test/mom.png --dad_path ./test/dad.png
```

## Training

### Train CAE

This is still in progress. Code lives in 
```
scripts\train_CAE.py
```


### Train Blender/DNA-Net+

This is still in progress. Code lives in 
```
scripts\train_blender.py
```
## Misc/eval scripts




## Credits
 - stylegan2: https://github.com/rosinality/stylegan2-pytorch  
 - pSp: https://github.com/eladrich/pixel2style2pixel  
 - InsightFace_Pytorch: https://github.com/TreB1eN/InsightFace_Pytorch  
 - dlib: http://dlib.net/face_landmark_detection.py.html  
 - Face alignment: https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5
 - stylegan2: https://github.com/rosinality/stylegan2-pytorch  
 - pSp: https://github.com/eladrich/pixel2style2pixel  
 - InsightFace_Pytorch: https://github.com/TreB1eN/InsightFace_Pytorch  
 - dlib: http://dlib.net/face_landmark_detection.py.html  
 - Face alignment: https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5

