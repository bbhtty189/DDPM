import torch
import torch.optim as optim
from torchvision import transforms
import cv2
import os
import numpy as np
import math

from tqdm.auto import tqdm as tqdm_auto
from dataset import dataset
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusion import diffusionmodel
from dataclasses import dataclass
from torchvision import datasets
from utils import plt_result


@dataclass
class train_cfg:
    # data
    img_size = 32
    root = r""
    weight_path = r""
    dataset_type = "MNIST"
    # diffusion
    timesteps = 1000
    sampling_timesteps = 1000
    beta_schedule = "linear"
    clip_sample = True
    Loss_schedule = "L2"
    # dataloader
    batch_size = 16
    shuffle = True
    num_workers = 4
    # optim
    lr = 2e-4
    # train
    epochs = 100
    lr_warmup_steps = 1500
    # eval
    bs = 1
    in_channels = 1

def train():

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    if os.path.exists("./logs") is False:
        os.makedirs("logs")

    if os.path.exists("./data") is False:
        os.makedirs("./data")

    cfg = train_cfg()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])])

    train_dataset = None
    if cfg.dataset_type == "MNIST":
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    elif cfg.dataset_type == "CIFAR-10":
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    elif cfg.dataset_type == "":
        assert cfg.root == "", "please input a dataset root"
        img_paths = [os.path.join(cfg.root, i) for i in os.listdir(cfg.root)]
        train_dataset = dataset(img_paths, train_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.batch_size,
                                                   shuffle=cfg.shuffle,
                                                   pin_memory=True,
                                                   num_workers=cfg.num_workers,
                                                   )

    # model
    model = diffusionmodel(cfg, in_channels=cfg.in_channels, out_channels=cfg.in_channels, channels=128, n_res_blocks=2, attention_levels=[],
                           channel_multipliers=[1, 2, 2, 2], n_heads=8, d_cond=32*32)

    if cfg.weight_path != "":
        weight = torch.load(cfg.weight_path)
        model.load_state_dict(weight)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    cfg.lr_warmup_steps = int((len(train_dataloader) * cfg.epochs) / 10)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.epochs),
    )

    train_loss_list = []
    for epoch in range(cfg.epochs):
        with tqdm_auto(range(len(train_dataloader))) as pbar:
            model = model.train()
            loss = 0
            for global_step, batch in zip(pbar, train_dataloader):

                img, label = batch
                img = img.to(device)

                loss_value = model(img)

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                pbar.update(1)
                loss += loss_value.item()
                logs = {"Loss": loss / (global_step + 1), "Epoch": epoch, "Lr": lr_scheduler.get_last_lr()[0]}
                pbar.set_postfix(**logs)
                global_step += 1

            loss = loss / (global_step)
            train_loss_list.append(math.floor(loss * 1000) / 1000)

            model = model.eval()
            with torch.no_grad():
                sample, preds = model.sample(cfg.bs, cfg.in_channels, device)
                img = np.zeros((int(cfg.img_size), int(model.timesteps / 50) * int(cfg.img_size), cfg.in_channels))
                num = 0
                for k in range(model.timesteps):
                    if (k + 1) % 50 == 0:
                        pred = preds[k].cpu().detach().squeeze(0).permute(1, 2, 0)
                        pred = (pred * 0.5 + 0.5) * 255
                        pred = pred.numpy()
                        if cfg.in_channels > 1:
                            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                        for num1 in range(cfg.img_size):
                            for num2 in range(cfg.img_size):
                                for num3 in range(cfg.in_channels):
                                    img[num1, num2 + num * cfg.img_size, num3] = pred[num1, num2, num3]
                        num += 1

                sample = sample.cpu().detach().squeeze(0).permute(1, 2, 0)
                sample = (sample * 0.5 + 0.5) * 255
                sample = sample.numpy()
                if cfg.in_channels > 1:
                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join("./logs", str(epoch)+ ".jpg"), img)

            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    plt_result(np.arange(cfg.epochs), train_loss_list, "训练损失")

if __name__ == "__main__":
    train()





