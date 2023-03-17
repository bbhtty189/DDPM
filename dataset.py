import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

class dataset(Dataset):

    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.transform(img)

        return img

if __name__ == "__main__":

    root = r"E:\datasets\flower_photos\daisy"
    img_paths = [os.path.join(root, i) for i in os.listdir(root)]
    print(img_paths[0])

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])

    train_dataset = dataset(img_paths, train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4,
                                                )
    print(train_dataloader)
    for img in train_dataloader:
        print(img.shape)

