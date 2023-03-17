import torch
import torch.nn as nn
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNetModel
import torch.nn.functional as F
import numpy as np
import math
from torchvision import transforms
from dataclasses import dataclass

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class diffusionmodel(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, channels, n_res_blocks, attention_levels, channel_multipliers, n_heads, d_cond):
        super(diffusionmodel, self).__init__()

        self.cfg = cfg

        # UNet
        self.unet = UNetModel(in_channels=in_channels, out_channels=out_channels, channels=channels, n_res_blocks=n_res_blocks,
                              attention_levels=attention_levels, channel_multipliers=channel_multipliers, n_heads=n_heads, d_cond=d_cond)

        #reference diffusiondet/detector.py: def __init__() & diffusers/schedulers/scheduling_ddpm.py: def __init__()
        # diffusion
        beta_start = 0.0001
        beta_end = 0.02
        self.one = torch.tensor(1.0)
        self.timesteps = int(cfg.timesteps)
        self.sampling_timesteps = default(cfg.sampling_timesteps, self.timesteps)
        assert self.sampling_timesteps <= self.timesteps

        self.img_size = self.cfg.img_size

        betas = None
        if cfg.beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        elif cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps
        self.ddim_sampling_eta = 1.

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def forward(self, x=None):

        if self.training:
            bs = x.shape[0]
            t = torch.randint(0, self.timesteps, (bs,), device=x.device, dtype=torch.long)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
            sample = self.forward_process(x, t, noise)
            pred_noise = self.unet(sample, t, None)

            loss = 0
            if self.cfg.Loss_schedule == "L2":
                loss = F.mse_loss(pred_noise, noise)
            elif self.cfg.Loss_schedule == "L1":
                loss = F.l1_loss(pred_noise, noise)
            return loss

    # reference https://github.com/abarankab/DDPM/blob/main/ddpm/diffusion.py: def sample()
    @torch.no_grad()
    def sample(self, batch, in_channels, device):
        shape = (batch, in_channels, self.img_size, self.img_size)

        sample = torch.randn(shape, device=device, dtype=torch.float32)  # x_t

        preds = [sample]
        for time in range(self.timesteps-1, -1, -1):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise = self.unet(sample, time_cond, None)

            sample = self.pred_prev_from_predNoise(sample, time_cond, pred_noise)

            preds.append(sample)

        return sample, preds

    # reference diffusers/schedulers/scheduling_ddpm.py: def add_noise()
    def forward_process(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)

        self.alphas_cumprod = self.alphas_cumprod.to(device=x.device, dtype=x.dtype)

        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = torch.sqrt(1. - self.alphas_cumprod[t])
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        sample = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

        return sample

    # reference diffusers/schedulers/scheduling_ddpm.py: def step()
    # reference https://github.com/abarankab/DDPM/blob/main/ddpm/diffusion.py:  def sample()
    @ torch.no_grad()
    def pred_prev_from_predNoise(self, sample, t, pred_noise):

        self.betas = self.betas.to(device=sample.device, dtype=sample.dtype)
        self.alphas = self.alphas.to(device=sample.device, dtype=sample.dtype)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device=sample.device, dtype=sample.dtype)

        # 1. compute alphas, betas
        betas_t = self.betas[t]
        alphas_t = self.alphas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        sqrt_recip_alphas_t = torch.sqrt(1 / alphas_t)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.flatten()
        while len(sqrt_recip_alphas_t.shape) < len(sample.shape):
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)

        coeff = betas_t / sqrt_one_minus_alphas_cumprod_t
        coeff = coeff.flatten()
        while len(coeff.shape) < len(sample.shape):
            coeff = coeff.unsqueeze(-1)

        # 2. remove noise
        mean = (sample - coeff * pred_noise) * sqrt_recip_alphas_t

        # 3. add noise
        variance = 0.
        if t > 0:
            variance_noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype)
            # reference diffusers/schedulers/scheduling_ddpm.py: def _get_variance
            # c = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
            # variance = c * variance_noise
            # variance = torch.clamp(variance, min=1e-20)

            # reference https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py#L188
            posterior_log_variance_clipped_t = self.posterior_log_variance_clipped[t]
            while len(posterior_log_variance_clipped_t.shape) < len(sample.shape):
                posterior_log_variance_clipped_t = posterior_log_variance_clipped_t.unsqueeze(-1)
            variance = torch.exp(0.5 * posterior_log_variance_clipped_t) * variance_noise

        pred_prev_sample = mean + variance

        return pred_prev_sample

if __name__ == "__main__":
    @dataclass
    class TrainingConfig:
        # data
        img_size = 32
        root = r"/root/autodl-tmp/CIFAR-10"
        weight_path = r""
        # diffusion
        timesteps = 1000
        sampling_timesteps = 1000
        beta_schedule = "cosine"
        clip_sample = True
        # dataloader
        batch_size = 16
        shuffle = True
        num_workers = 4
        # optim
        lr = 1e-4
        # train
        epochs = 100
        lr_warmup_steps = 1500

    cfg = TrainingConfig()

    model = diffusionmodel(cfg, device=torch.device("cuda:0"), in_channels=3, out_channels=3, channels=64, n_res_blocks=2,
                           attention_levels=[1,3], channel_multipliers=[1,2,4,8], n_heads=8, d_cond=512*512).to("cuda:0")

    img_path = r"E:\datasets\COCO\val2017\000000000285.jpg"

    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    img = transform(img)
    img = img.unsqueeze(0)
    print(img.shape)

    model.train(False)
    img = img.to("cuda:0")
    timestep = 1000
    bs = img.shape[0]
    t = torch.randint(0, timestep, (bs,), device=img.device)

    x, preds = model(img)
    print(x.shape)













