import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FoggyDataset(Dataset):
    def __init__(self, foggy_dir, gt_dir, image_size=(256, 256)):
        self.foggy_dir = foggy_dir
        self.gt_dir = gt_dir
        self.image_size = image_size

        self.foggy_images = os.listdir(foggy_dir)
        self.gt_images = os.listdir(gt_dir)

        if len(self.foggy_images) != len(self.gt_images):
            raise ValueError("Number of foggy and ground truth images do not match")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.foggy_images)

    def __getitem__(self, idx):
        foggy_path = os.path.join(self.foggy_dir, self.foggy_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        foggy_img = Image.open(foggy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        foggy_tensor = self.transform(foggy_img)
        gt_tensor = self.transform(gt_img)

        return foggy_tensor, gt_tensor
        