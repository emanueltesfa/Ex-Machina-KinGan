import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Resize
from utils.utils import align_face, totensor
from torch.utils.data import Dataset
import PIL
class FiW_Dataset(Dataset):
    def __init__(self, triplets_file, img_dir, align_faces=False, device=None):
        self.triplets = pd.read_csv(triplets_file)
        self.img_dir = img_dir
        self.align_faces = align_faces
        self.resize = Resize((256,256)) # hard coded from inference img shape
        self.device = device

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        filenames = list(self.triplets.iloc[idx,1:])

        images = []
        # triplets stores as father, mother, child
        for filename in filenames:
            img_path = os.path.join(self.img_dir, filename)
            image = None
            if self.align_faces:
                try:
                    image = align_face(img_path).convert('RGB')
                except UnboundLocalError:
                    image = PIL.Image.open(img_path).convert('RGB') #
                image = totensor(image).to(self.device)
            else:
                image = self.resize(read_image(img_path).float())
            if self.device:
                images.append(image.to(self.device))
            

        return images