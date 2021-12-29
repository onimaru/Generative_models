import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch import nn
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(torch.nn.Module):
    def __init__(self,X_dim,h_dim,Z_dim):
        super(Encoder,self).__init__()
        self.hidden1 = torch.nn.Linear(X_dim       , X_dim)
        self.hidden2 = torch.nn.Linear(X_dim       , h_dim)
        self.hidden3 = torch.nn.Linear(h_dim       , int(h_dim/2))
        self.hidden4 = torch.nn.Linear(int(h_dim/2), int(h_dim/4))
        self.out1    = torch.nn.Linear(int(h_dim/4), Z_dim)
        self.out2    = torch.nn.Linear(int(h_dim/4), Z_dim)
    
    def forward(self,X):
        h = nn.Dropout(p=0.0)(F.selu(self.hidden1(X)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden2(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden3(h)))
        h = nn.Dropout(p=0.0)(F.selu(self.hidden4(h)))
        mu      = self.out1(h)
        log_var = self.out2(h)
        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self,Z_dim,h_dim,X_dim):
        super(Decoder,self).__init__()
        self.hidden1 = torch.nn.Linear(Z_dim       , int(h_dim/4))
        self.hidden2 = torch.nn.Linear(int(h_dim/4), int(h_dim/2))
        self.hidden3 = torch.nn.Linear(int(h_dim/2), h_dim)
        self.hidden4 = torch.nn.Linear(h_dim       , X_dim)
        self.out     = torch.nn.Linear(X_dim       , X_dim)
    
    def forward(self,z):
        h   = nn.Dropout(p=0.0)(F.selu(self.hidden1(z)))
        h   = nn.Dropout(p=0.0)(F.selu(self.hidden2(h)))
        h   = nn.Dropout(p=0.0)(F.selu(self.hidden3(h)))
        h   = nn.Dropout(p=0.0)(F.selu(self.hidden4(h)))
        out = torch.sigmoid(self.out(h))
        return out
    
class VAE(torch.nn.Module):
    def __init__(self,X_dim,h_dim,Z_dim):
        super(VAE,self).__init__()
        self.X_dim   = X_dim
        self.h_dim   = h_dim
        self.Z_dim   = Z_dim
        self.encoder = Encoder(self.X_dim,self.h_dim,self.Z_dim).to(device)
        self.decoder = Decoder(self.Z_dim,self.h_dim,self.X_dim).to(device)
        
    def sample_z(self,mu, log_sigma):
        z = Variable(torch.randn(mu.shape[0], self.Z_dim)).to(device)
        return mu + torch.exp(log_sigma / 2) * z
    
    def init_optimizer(self,lr):
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        
    def init_weights(self):
        def _init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight).to(device)
                m.bias.data.fill_(0.001)
        self.encoder.apply(_init_weights)
        self.decoder.apply(_init_weights)
        
    def init_vae(self,lr):
        self.init_optimizer(lr)
        self.init_weights()
        
    def blank_part_of_img(self,X,size,random=False):
        blanked_img = X.view(size,1,28,28).clone()
        blank_size = np.random.randint(5,15)
        if random == True:
            blank_indexes = np.random.randint(0,14,size=2)
        else:
            blank_indexes = np.array([0,0])
        blanked_img[
            :,:,
            blank_indexes[0]:blank_indexes[0]+blank_size,
            blank_indexes[1]:blank_indexes[1]+blank_size
        ] = 0.0
        return blanked_img.view(-1,self.X_dim)
        
    def train(self,epochs,loader,lr,verbose=1):
        self.init_vae(lr)
        self.loss_his = []
        for epoch in range(1,epochs+1):
            for step, batch_x in enumerate(loader):
                z = Variable(torch.randn(loader.batch_size, self.Z_dim)).to(device)
                X = Variable(batch_x[0].view(-1,self.X_dim)).to(device)
                if np.random.rand() < 0.01:
                    X_blank = self.blank_part_of_img(X,loader.batch_size,random=True)
                    z_mu, z_var = self.encoder(X_blank)
                else:
                    z_mu, z_var = self.encoder(X)
                z = self.sample_z(z_mu, z_var)
                X_sample = self.decoder(z)
                recon_loss = F.binary_cross_entropy(X_sample, X,reduction='sum') / X.shape[0]
                kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
                loss = recon_loss + kl_loss 
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.loss_his.append(loss.data)
            if epoch % 10 == 0 and verbose == 1:
                print('Epoch-{}| Average loss: {:.5f}'.format(epoch, loss.mean().data))
    
    def evaluate(self,X):
        X           = X.view(-1,self.X_dim)
        z_mu, z_var = self.encoder(X.to(device))
        z           = self.sample_z(z_mu.detach(), z_var.detach()).detach()
        X_sample    = self.decoder(z).detach()
        recon_loss  = F.binary_cross_entropy(X_sample, X,reduction='sum') / X.shape[0]
        kl_loss     = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        return {"reconstruction_error":recon_loss.detach().item(),"kl_divergence":kl_loss.detach().item()}
    
    def evaluate_loader(self,loader):
        errors = {"reconstruction_error":[],"kl_divergence":[]}
        recon_loss_hist = []
        kl_loss_hist = []
        for step, batch_x in enumerate(loader):
            X = batch_x[0].view(-1,self.X_dim).to(device)
            z_mu, z_var = self.encoder(X)
            z = self.sample_z(z_mu.detach(), z_var.detach()).detach()
            X_sample = self.decoder(z).detach()
            recon_loss = F.binary_cross_entropy(X_sample, X,reduction='sum') / X.shape[0]
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
            recon_loss_hist.append(recon_loss.detach().item())
            kl_loss_hist.append(kl_loss.detach().item())
        return {"reconstruction_error":np.mean(recon_loss_hist),"kl_divergence":np.mean(kl_loss_hist)}
    
    def encode(self,X):
        X = X.view(-1,self.X_dim)
        mu,log_sigma = self.encoder(X.to(device))
        return self.sample_z(mu,log_sigma)
    
    def decode(self,Z):
        return self.decoder(Z.to(device))
        
    def identity(self,X):
        X = X.view(-1,self.X_dim)
        return self.decode(self.encode(X.to(device)))