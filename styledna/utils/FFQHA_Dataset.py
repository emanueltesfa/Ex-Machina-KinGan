import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Resize
from utils.utils import align_face, totensor
from torch.utils.data import Dataset
class FFQHA_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, align_faces=False):
        self.filenames = next(os.walk(img_dir), (None, None, []))[2]  # [] if no file
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.align_faces = align_faces
        self.resize = Resize((256,256)) # hard coded from inference img shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_num, _ = os.path.splitext(filename)
        img_num = int(img_num)

        img_path = os.path.join(self.img_dir, filename)
        
        if self.align_faces:
            image = align_face(img_path).convert('RGB')
            image = totensor(image).unsqueeze(0)
        else:
            image = self.resize(read_image(img_path).float())

        age_range = self.img_labels.loc[img_num,"age_group"].split('-')
        age = int((int(age_range[0]) + int(age_range[1])) / 2)
        gender = self.img_labels.loc[img_num,"gender"]
        gender = int(gender == 'male')
        label = (age,gender)


        return image, label