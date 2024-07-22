import torch
import torchvision
import matplotlib.pyplot as plt
import os

from torchvision.datasets import Image Folder
from torchvision import transforms
import os
import random

def show_images (data, num_samples=20, cols=4): 

    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
         img, data[random.randint(0, len(data)-1)] 
         plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
         plt.imshow(img)
         plt.axis('off')
     plt.show()


import torch.nn.functional as F
def linear_beta_schedule(timesteps, start-0.0001, end=0.02): 
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch size t.shape[0]
    out vals.gather(-1, t.cpu()) 
    return out.reshape(batch_size, ((1,) (len(x_shape) 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise torch.randn_like(x_0) 
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.sh sqrt_one_minus_alphas_cumprod_t = get_index_from_list( ) sqrt_one_minus_alphas_cumprod, t, x_0.shape)
#mean variance
    return sqrt_alphas_cumprod_t.to(device) x_0.to(device) \ + sqrt_one_minus_alphas_cumprod_t.to(device) noise.to (device), noise.to(

#Define beta schedule
T= 300
betas linear_beta_schedule(timesteps=T)

#Pre-calculate different terms for closed form
alphas 1. betas
alphas_cumprod torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas torch.sqrt(1.0/ alphas)
sqrt_alphas_cumprod torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod torch.sqrt(1. alphas_cumprod)
posterior_variance betas (1. alphas_cumprod_prev) / (1. alphas_cumprod)

from torchvision import transforms from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

IMG SIZE = 64
BATCH SIZE 128
data_transform transforms.Compose([ 
  transforms.Resize((IMG_SIZE, IMG_SIZE)), 
  transforms.RandomHorizontalFlip(), 
  transforms.ToTensor(), # Convert PIL image to tensor 
  transforms.Lambda (lambda t: (t2)-1) # Scale between [-1, 1]
])

#Load dataset

data_dir r"C:\Users\SEJAl\Downloads\cars\cars_test"
data torchvision.datasets. ImageFolder(root=data_dir, transform=data_transfor
dataloader DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=T

#Example of accessing images from Dataloader
for images, in dataloader:
plt.figure(figsize=(15, 15))
plt.axis('off')
num_images = 10
stepsize int(len(images) / num_images) for idx in range(0, len(images), stepsize):
img images[idx]
plt.subplot(1, num_images + 1, int(idx / stepsize) + 1) plt.imshow(transforms. ToPILImage() (img)) #Convert tensor to PIL imag
plt.show()
break


from torch import nn
import math

class Block(nn.Module):
  def _init(self, in_ch, out_ch, time_emb_dim, up=False):
    super().init()
    self.time_mlp nn. Linear(time_emb_dim, out_ch)
    if up:
      self.conv1 nn.Conv2d(2*in_ch, out_ch, 3, padding=1) 
      self.transform nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 nn.Conv2d(in_ch, out_ch, 3, padding=1) 
      self.transform nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    self.conv2 nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.bnormi nn. BatchNorm2d(out_ch)
    self.bnorm2 nn. BatchNorm2d(out_ch)
    self.relu = nn.ReLU()

  def forward(self, x, t, ):
#First Conv
     h = self.bnorm1(self.relu(self.conv1(x)))
#Time embedding
     time_emb self.relu(self.time_mlp(t))
#Extend Last 2 dimensions
     time_emb = time_emb[(...,) + (None,) 2]
#Add time channel
     h=h+time_emb
#Second Conv
     h = self.bnorm2(self.relu(self.conv2(h))
                     #Down or Upsample
     return self.transform(h)
class SinusoidalPositionEmbeddings (nn.Module):
  def init (self, dim): 
    super().init_()
    self.dim dim

  def forward(self, time):
    device time.device 
    half dimself.dim // 2 
    embeddings math.log(10000) / (half_dim1) 
    embeddings torch.exp(torch.arange(half_dim, device=device)*-embedding
    embeddings time[:, None] embeddings [None,:] 
    embeddings torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) #TODO: Double check the ordering here
    return embeddings

class Simplelnet (nn.Module):
  def init (self):
    super().init()
    Image channels = 3
    down_channels (64, 128, 256, 512, 1024) up_channels (1024, 512, 256, 128, 64)
    out dim = 3
    time_emb_dim = 32

#Time embedding
    self.time_mlp nn.Sequential( 
      Sinusoidal PositionEmbeddings (time_emb_dim), 
      nn. Linear(time_emb_dim, time_emb_dim), 
      nn.ReLU()
      )

#Initial projection
    self.convenn.Conv2d(image_channels, down_channels[0], 3, padding=1)

# Downsample
    self.downs nn. ModuleList([Block(down_channels[i], down_channels [1+1] time_emb_dim) \ 
                               for i in range(len(down_channels)-1)])

#Upsample
    self.ups nn.ModuleList([Block(up_channels [1], up_channels [1+1], \ time emb dim, up=True) \ 
                            for i in range(len(up_channels)-1)])

#Edit: Corrected a bug found by Jakub C (see YouTube comment)
    self.output nn.Conv2d(up_channels [-1], out_dim, 1)

  def forward(self, x, timestep):
#Embedd time
    t = self.time_mlp(timestep)
#Initial conv
    x = self.conve(x)
#Unet
    residual_inputs = []
    for down in self.downs:
        x = down(x, t)
        residual_inputs.append(x)
    for up in self.ups:
        residual_x residual_inputs.pop()

#Add residual x as additional channels
        x = torch.cat((x, residual_x), dim=1)
        x = x up(x, t)
    return self.output(x)
model SimpleUnet()
print("Num params:", sum(p.numel() for p in model.parameters()))

def sample timestep(x, t):
        betas_t get_index_from_list(betas, t, x.shape).
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
          sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t= get_index_from_list(sqrt_recip_alphas, t, x.shape)
        model_mean sqrt_recip_alphas_t *(
          betas t model(x, t) / sqrt_one_minus_alphas_cumprod_t
        ) 
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if t == 0:
#As pointed out by Luis Pereira (see YouTube comment)
#The t's are offset from the t's in the paper
        return model_mean
    else:
         noise torch.randn_like(x)
         return model mean torch.sqrt(posterior_variance_t) noise

def sample_plot_image():

#Sample noise
  img size IMG_SIZE
  ing torch.randn((1, 3, img size, img size), device=device)
  plt.figure(figsize=(15,15))
  plt.axis('off')
  num_images 10
  stepsize int(T/num_images)
  for i in range(e,T) [::-1]:
      t = torch.full((1,), i, device=device, dtype=torch.long)
      img = sample_timestep(img, t)
      img = torch.clamp(img, 1.0, 1.0)
  if 1% stepsize = 0:
    plt.subplot(1, num_images, int(1/stepsize)+1)
    show_tensor_image(img.detach().cpu())
  plt.show()
