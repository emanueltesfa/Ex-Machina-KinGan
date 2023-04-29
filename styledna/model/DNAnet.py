import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.init as init
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #init.constant_(m.weight.data, 0)
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
            #init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


class DNAnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                weights_init(m)

    def forward(self, m, f, alpha=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        m_gene = self.encoder(m)
        f_gene = self.encoder(f)
        # alpha is 1 dimensional guassian/normal 
        alpha = torch.randn(1, dtype=torch.float32).to(device)

        if alpha is not None:
            print(alpha.is_cuda, m_gene.is_cuda, f_gene.is_cuda )
            print(m_gene.dtype,m_gene.shape, f_gene.dtype, f_gene.shape)
            s_gene = alpha * m_gene + (1 - alpha) * f_gene
        else:
            s_gene = torch.max(
                torch.cat((m_gene.unsqueeze(0), f_gene.unsqueeze(0)), 0), dim=0)[0]
            if m.size(0) != 1:
                s_gene = s_gene.squeeze(0)
        s = self.decoder(s_gene)
        return s


def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
    """Computes KL[p||q] between two Gaussians defined by [mu, log_sigma]."""
    return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
                       / (2 * torch.exp(log_sigma2) ** 2) - 0.5


class VAE(nn.Module):
    def __init__(self, nz, beta=0.25, device='cpu'):
        super().__init__()
        self.beta = beta # factor trading off between two loss components
      
        # self.encoder = Encoder(nz*2,input_size=in_size).to(device)
        # self.decoder = Decoder(nz,output_size=in_size).to(device)
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, nz*2),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(nz, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
        ).to(device)
        self.nz = nz

    def forward(self, x):
        code = self.encoder(x)
        n = code.shape[0]
        mu,sig = code.split(self.nz,dim=1)
        sample = torch.distributions.MultivariateNormal(torch.zeros(self.nz),torch.eye(self.nz)).sample((n,)).to(self.device)
        z = mu + (sig*sample)
        
        q = code
        reconstruction = self.decoder(z)

        return {'q': q, 
                    'rec': reconstruction}

    def loss(self, x, outputs):
        mse = torch.nn.MSELoss()

        code = outputs["q"]
        rec = outputs["rec"].squeeze(1)
        mu, sig = code.split(self.nz,dim=1)
        b = mu.shape[0]
        rec_loss = mse(x,rec)
        prior_mu = torch.zeros(b,self.nz).to(self.device)
        prior_sig = torch.ones(b,self.nz).to(self.device)
        kl_loss = kl_divergence(mu.to(self.device),sig.to(self.device),prior_mu,prior_sig)

        # return weighted objective
        return rec_loss + self.beta * kl_loss, \
                    {'rec_loss': rec_loss, 'kl_loss': kl_loss}
      
    def reconstruct(self, x):
        """Use mean of posterior estimate for visualization reconstruction."""
        code = self.encoder(x.to(self.device))
        mu,sig = code.split(self.nz,dim=1) if len(code.shape) > 1 else code.split(self.nz)
        image_flat = self.decoder(mu.to(self.device))
        image = image_flat.reshape(1,28,28)
        return image

if __name__ == '__main__':
    net = DNAnet().to('cuda')
    m = torch.randn(3, 512).to('cuda')
    f = torch.randn(3, 512).to('cuda')
    s = net(m, f)
    print(s.size())