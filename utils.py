import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    airplane_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    airplane_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(airplane_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in airplane_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(airplane_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(airplane_class)), airplane_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_three_data(root: str, train_val_rate: float = 0.8, train_rate: float = 0.8):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    test_path = r"/tmp/pycharm_project_728/DiffClass/test"
    if os.path.exists(test_path) is False:
        os.makedirs(test_path)
    # 存放测试数据
    f_test = open(os.path.join(test_path, 'test.txt'), 'w')

    # 遍历文件夹，一个文件夹对应一个类别
    airplane_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    airplane_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(airplane_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []
    test_images_label = []
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    # 按每一个类别划分数据，分别放入训练集，验证集，测试集
    for cla in airplane_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        # 先获得训练验证集
        train_val_path = random.sample(images, k=int(len(images) * train_val_rate))
        # 不在训练验证集中的样本放入测试集
        for img_path in images:
            if img_path not in train_val_path:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
                in_str = img_path + " " + str(image_class)
                f_test.write(in_str + '\n')

        train_path = random.sample(train_val_path, int(len(train_val_path) * train_rate))
        for img_path in train_val_path:
            if img_path in train_path:  # 如果该路径在采样的验证集样本中则存入验证集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
            else:
                val_images_path.append(img_path)
                val_images_label.append(image_class)

    f_test.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(airplane_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(airplane_class)), airplane_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

# 开始训练
def train_one_epoch(model, decoder, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = F.interpolate(pred, size=32, mode="bilinear", align_corners=False)
        pred = F.interpolate(pred, size=20, mode="bilinear", align_corners=False)
        pred = decoder(pred)

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, decoder, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = F.interpolate(pred, size=32, mode="bilinear", align_corners=False)
        pred = F.interpolate(pred, size=20, mode="bilinear", align_corners=False)
        pred = decoder(pred)

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 开始训练
def train_one_epoch_unet(model, decoder, optimizer, diffusion, optimizer_unet, DDPM_scheduler, DDIM_scheduler, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    optimizer_unet.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        # pred = F.interpolate(pred, size=32, mode="bilinear", align_corners=False)

        noise = torch.randn(pred.shape).to(pred.device)
        bs = pred.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, DDPM_scheduler.num_train_timesteps, (bs,), device=pred.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = DDPM_scheduler.add_noise(pred, noise, timesteps)

        noise_pred = diffusion(noisy_images, timesteps, return_dict=False)[0]
        unet_loss = F.mse_loss(noise_pred, noise)

        unet_loss.backward()
        optimizer_unet.step()
        optimizer_unet.zero_grad()

        sample_noise = torch.randn(pred.shape).to(pred.device)
        sample = sample_noise

        for i, t in enumerate(DDIM_scheduler.timesteps):
            # 1. predict noise residual
            with torch.no_grad():
                residual = diffusion(sample, t).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = DDIM_scheduler.step(residual, t, sample).prev_sample

        pred = F.interpolate(sample, size=20, mode="bilinear", align_corners=False)
        pred = decoder(pred)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss = loss.detach() + unet_loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_unet(model, decoder, diffusion, DDPM_scheduler, DDIM_scheduler, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        # pred = F.interpolate(pred, size=32, mode="bilinear", align_corners=False)

        noise = torch.randn(pred.shape).to(pred.device)
        bs = pred.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, DDPM_scheduler.num_train_timesteps, (bs,), device=pred.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = DDPM_scheduler.add_noise(pred, noise, timesteps)

        noise_pred = diffusion(noisy_images, timesteps, return_dict=False)[0]
        unet_loss = F.mse_loss(noise_pred, noise)

        sample_noise = torch.randn(pred.shape).to(pred.device)
        sample = sample_noise

        for i, t in enumerate(DDIM_scheduler.timesteps):
            # 1. predict noise residual
            with torch.no_grad():
                residual = diffusion(sample, t).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = DDIM_scheduler.step(residual, t, sample).prev_sample

        pred = F.interpolate(sample, size=20, mode="bilinear", align_corners=False)
        pred = decoder(pred)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss = loss + unet_loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def plt_result(x, y, title):

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.title(title+"结果")
    plt.xlabel("epochs")
    plt.ylabel(title)

    plt.plot(x, y, marker='o', color='blue', markerfacecolor='yellow')

    for a, b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=8)

    plt.savefig(os.path.join("./logs", title+".jpg"))
    # plt.show()

# 读取测试集
def load_test_data(test_path):
    with open(test_path, "r") as f:
        items = f.readlines()

    test_images_path = []
    test_images_label = []
    for item in items:
        data = item.split()
        test_images_path.append(data[0])
        test_images_label.append(int(data[1]))

    return test_images_path, test_images_label


