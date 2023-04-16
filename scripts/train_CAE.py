import torch
import os
import argparse

from torchvision.utils import save_image
from model.model import pSp, condi
from model.DNAnet import DNAnet
from utils.utils import align_face, totensor
from utils.FFQHA_Dataset import FFQHA_Dataset
import torch.nn as nn
import PIL

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    print("Loading models...")
    # # Load model weights
    # DNA = DNAnet().to(device)
    # DNA.load_state_dict(torch.load(args.dna), strict=True)
    # DNA.eval()

    mapper = nn.DataParallel(condi()).to(device)
    mapper.load_state_dict(torch.load(args.map), strict=True)
    mapper.eval()

    net = nn.DataParallel(pSp(3, args.enc, args.gan)).to(device)

    batch_size = 1
    num_epochs = 1
    training_data = FFQHA_Dataset(args.anno,args.ffqh)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), 0.0001)
    CoSim = torch.nn.CosineSimilarity()

    E = net.module.encoder # encoder

    def D(W, age, gender):
        sW_hat_expand = W.repeat(18, 1, 1).permute(1, 0, 2)
        print("here",sW_hat_expand.shape, age.shape,  gender.shape)
        sW_hat_delta = mapper(sW_hat_expand, age,  gender)
        sImg_hat = net(sW_hat_expand + sW_hat_delta)
        return sImg_hat

    

    for epoch in range(num_epochs):
        for i, (img,a_g) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            age = torch.randint(0,100,size=[batch_size,1]).to(device)
            gender = torch.bernoulli(torch.ones(batch_size,1)*0.5).to(device)
            
            W_hat = E(img.to(device))
        
            age_orig, gender_orig = a_g  
            age_orig = age_orig.to(device).reshape(-1,1)
            gender_orig = gender_orig.to(device).reshape(-1,1)

            Igen = D(W_hat, age, gender)
            Irec = D(W_hat, age_orig, gender_orig)
            Icyc = D(E(Igen), age_orig, gender_orig)


            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')

    # Image Path and Saving Path
    parser.add_argument('-m', '--mom_path', help='Path of mom image', default = r'C:\Users\tjtom\OneDrive\Desktop\File_Cabinet\Code\Ex-Machina-GAN\mom.png')
    parser.add_argument('-d', '--dad_path', help='Path of dad image')
    parser.add_argument('-a', '--target_gender', help='Predicted child age', default= 'male')
    parser.add_argument('-g', '--target_age', help='Predicted child gender', default = 15)
    parser.add_argument('-o', '--outputs', help='Output directory', default='./result')

    # Model weights Path
    parser.add_argument('-E', '--enc', help='Path of pretrained encoder model', default='./pretrained_model/enc_4_2.pth')
    parser.add_argument('-M', '--map', help='Path of pretrained mapper model', default='./pretrained_model/condi_4_2.pth')
    parser.add_argument('-G', '--gan', help='Path of pretrained stylegan2 model', default='./pretrained_model/stylegan2-ffhq-config-f.pt')
    parser.add_argument('-D', '--dna', help='Path of pretrained DNA model', default='./pretrained_model/10.pth')

    parser.add_argument('-A', '--anno', help='Path of FFQH Aging Dataset Annotations', default='FFHQ-Aging-Dataset/ffhq_aging_labels.csv')
    parser.add_argument('-FF', '--ffqh', help='Path of FFQH Aging Data', default='FFHQ-Aging-Dataset/data')

    # Device
    parser.add_argument('--device', help='Device to be used by the model (default=cuda:0)', default="cuda")
    args = parser.parse_args()

    ### RUN
    main(args)