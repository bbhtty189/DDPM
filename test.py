import torch
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

from diffusion import diffusionmodel

@dataclass
class predict_cfg:
    # data
    img_size = 32
    weight_path = r"./weights/model-99.pth"
    # diffusion
    timestep = 1000
    sampling_timestep = 100


def predict():
    cfg = predict_cfg()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = diffusionmodel(cfg, device, in_channels=3, out_channels=3, channels=64, n_res_blocks=2,
                           attention_levels=[1, 3], channel_multipliers=[1, 2, 4, 8], n_heads=8, d_cond=32*32)

    if cfg.weight_path != "":
        weight = torch.load(cfg.weight_path)
        model.load_state_dict(weight)

    model = model.to(device)
    for i in range(10000):
        print(f"the {i+1} photo is generated")
        model = model.eval()
        with torch.no_grad():
            x0, preds = model(None)

            img = np.zeros((int(cfg.img_size), int(model.sampling_timesteps) * int(cfg.img_size), 3))
            for k in range(model.sampling_timesteps):
                pred = preds[k].cpu().squeeze().permute(1, 2, 0)
                # print(pred.shape)
                pred = (pred * 0.5 + 0.5) * 255
                pred = pred.numpy()
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                for num1 in range(cfg.img_size):
                    for num2 in range(cfg.img_size):
                        for num3 in range(3):
                            img[num1, num2 + k * cfg.img_size, num3] = pred[num1, num2, num3]

            x0 = x0.cpu().squeeze().permute(1, 2, 0)
            x0 = (x0 * 0.5 + 0.5) * 255
            x0 = x0.numpy()
            x0 = cv2.cvtColor(x0, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(r"E:\datasets\CIFAR-predict", str(i) + ".jpg"), img)

if __name__ == "__main__":
    predict()

