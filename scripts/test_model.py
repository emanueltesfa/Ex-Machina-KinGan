import torch
import torch.nn
import os
import argparse

from torchvision.utils import save_image
from model.model_irse import Backbone
from model.model_arcface import *
import pandas as pd
from PIL import Image
import numpy as np

from model.model import pSp, condi
from model.DNAnet import DNAnet
from utils.utils import align_face, totensor
import torch.nn as nn
import torchvision
### Used to create scores for the roc plots
def infer(fimg, mimg, net, DNA, mapper, device="cuda:0"):
    # Inference
    
    mImg = align_face(fimg).convert('RGB')
    dImg = align_face(mimg).convert('RGB')

    testAge =  torch.ones((1,1)) * (args.age) /100 
    if args.gender == 'male':
        testGen = torch.ones(1, 1).to(device)
    else:
        testGen = torch.zeros(1, 1).to(device)

    with torch.no_grad():
        #print(mImg)
        #print(dImg)
        mImg = totensor(mImg).unsqueeze(0).to(device)
        dImg = totensor(dImg).unsqueeze(0).to(device)
        x = net.module.encoder(mImg)
        print(x.shape)
        sW_hat = DNA(net.module.encoder(mImg), net.module.encoder(dImg))
        sW_hat_expand = sW_hat.repeat(18, 1, 1).permute(1, 0, 2)
        sW_hat_delta = mapper(sW_hat_expand, testAge,  testGen)
        sImg_hat = net(sW_hat_expand + sW_hat_delta)
        sImg_hat = torch.nn.functional.interpolate(sImg_hat, size=(112,112),mode='bicubic')
                
    return ((sImg_hat+1)/2).clip(0,1)
def save_img(img,filename):
    img = torchvision.transforms.functional.to_pil_image(img)
    img.save(filename)
                
def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # Load model weights
    DNA = DNAnet().to(device)
    DNA.load_state_dict(torch.load(args.dna), strict=True)
    DNA.eval()

    net = nn.DataParallel(pSp(3, args.enc, args.gan)).to(device)
    net.eval()

    mapper = nn.DataParallel(condi()).to(device)
    mapper.load_state_dict(torch.load(args.map), strict=True)
    mapper.eval()

    arcface = Backbone(num_layers=50,drop_ratio=0.6,mode="ir_se").to(device)
    arcface.load_state_dict(torch.load(args.irse50))
    arcface.eval()

    triplets = pd.read_csv(args.triplets)
    triplets = triplets.to_numpy()[1:,1:]


    
    print(triplets.shape)
    
    scores = []
    cos = torch.nn.CosineSimilarity(dim=0).to(device)
    i = 0
    with torch.no_grad():
        for i in range(1000):
            try:
                print(f"{i}/1000")
                n = np.random.randint(low=0,high=triplets.shape[0])
                e = np.random.randint(low=0,high=triplets.shape[0])
                triplet = triplets[n]
                triplet[2] = triplets[e][2]
                triplet_images = []
                for path in triplet:
                    img = Image.open(path).resize((112,112))
                    arr = np.array(img)
                    arr = torch.tensor(arr).permute(2,0,1).float() / 255
                    triplet_images.append(arr)
                triplet_tensors = torch.stack(triplet_images).to(device)

                p1, p2, _ = triplet
                out = infer(p1,p2,net,DNA,mapper,device)
                p1_hat,p2_hat,c_hat = arcface(triplet_tensors)
                # import pdb; pdb.set_trace()
                out_hat = arcface(out).reshape(-1)
                # save_img(triplet_tensors[0],f"tests\\father_{i}.png")
                # save_img(triplet_tensors[1],f"tests\\mother_{i}.png")
                # save_img(triplet_tensors[2],f"tests\\child_{i}.png")
                # save_img(out.squeeze(0),f"tests\\rec_{i}.png")
                p1_score = cos(p1_hat,c_hat).item()
                p2_score = cos(p2_hat,c_hat).item()

                # import pdb; pdb.set_trace()
                p1_rec_score = cos(p1_hat,out_hat).item()
                p2_rec_score = cos(p2_hat,out_hat).item()
                pairwise = cos(c_hat,out_hat).item()

                # import pdb; pdb.set_trace()
                scores.append([p1_score,p2_score,p1_rec_score,p2_rec_score,pairwise])
            except UnboundLocalError:
                print("fail!")
                continue
            if i % 50 == 0: 
                scores_save = np.array(scores)
                pd.DataFrame(scores_save).to_csv(f"fake_scores_{i}.csv")

    scores = np.array(scores)
    pd.DataFrame(scores).to_csv("fake_scores_.csv")
    import pdb; pdb.set_trace()
    


        
        
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')
    parser.add_argument('-IR', '--irse50', help='IRSE50 pretrained model weights', default='./pretrained_model/model_ir_se50.pth')
    parser.add_argument('-T', '--triplets', help='file where father, mother, child triplets are stored. obtained from generate_fiw_triplets.py', default='triplets.csv')

    # Model weights Path
    parser.add_argument('-E', '--enc', help='Path of pretrained encoder model', default='./pretrained_model/enc_4_2.pth')
    parser.add_argument('-M', '--map', help='Path of pretrained mapper model', default='./pretrained_model/condi_4_2.pth')
    parser.add_argument('-G', '--gan', help='Path of pretrained stylegan2 model', default='./pretrained_model/stylegan2-ffhq-config-f.pt')
    parser.add_argument('-D', '--dna', help='Path of pretrained DNA model', default='./pretrained_model/10.pth')
    
    parser.add_argument('-A', '--age', help='target age', default=25)
    parser.add_argument('-GN', '--gender', help='target gender', default='male')


    args = parser.parse_args()

    ### RUN
    main(args)