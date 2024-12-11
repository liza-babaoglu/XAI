import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from medmnist import INFO
from medmnist.dataset import PathMNIST

from torchvision import models, transforms

data_flag = "pathmnist"
info = INFO[data_flag]
num_classes = len(info["label"])


transform = transforms.Compose([
    transforms.ToTensor()
])


test_dataset = PathMNIST(split='test', transform=transform, download=True,size=224)


class RISE:
    def __init__(self, model, input_size, num_masks=5000, mask_size=8):
        """
        初始化 RISE 解释器
        :param model: 待解释的模型
        :param input_size: 输入图像的大小 (H, W)
        :param num_masks: 随机遮罩的数量
        :param mask_size: 遮罩的低分辨率大小
        """
        self.model = model
        self.input_size = input_size
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.masks = self._generate_masks()

    def _generate_masks(self):
        """
        生成随机遮罩，尺寸为 (num_masks, H, W)
        """
        masks = np.random.choice([0, 1], size=(self.num_masks, self.mask_size, self.mask_size), p=[0.5, 0.5])
        masks = masks.astype(np.float32)
        masks = np.array([
            cv2.resize(mask, self.input_size, interpolation=cv2.INTER_LINEAR) for mask in masks
        ])
        return torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # (num_masks, 1, H, W)

    def explain(self, image, target_class):
        """
        生成 RISE 热图
        :param image: 输入图像 (C, H, W)
        :param target_class: 目标类别
        :return: 热图 (H, W)
        """
        image = image.unsqueeze(0)
        batch_size = 32
        heatmap = torch.zeros(self.input_size, dtype=torch.float32)


        for i in tqdm(range(0, self.num_masks, batch_size)):
            batch_masks = self.masks[i:i + batch_size]
            masked_images = image * batch_masks.cuda()
            outputs = self.model(masked_images).softmax(dim=1)
            scores = outputs[:, target_class].detach().cpu()


            for j, mask in enumerate(batch_masks):
                heatmap += mask.squeeze(0) * scores[j]


        heatmap /= self.num_masks
        return heatmap.numpy()


image, label = test_dataset[0]
vgg16 = models.vgg16(pretrained=False)
vgg16.classifier[5] = nn.Dropout(0.5)
vgg16.classifier[6] = torch.nn.Linear(4096, num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.load_state_dict(torch.load("best_vgg16_multiclass.pth"))
vgg16 = vgg16.to(device).eval()


input_image = image.to(device)


rise = RISE(vgg16, input_size=(224, 224), num_masks=5000, mask_size=8)
target_class = label
heatmap = rise.explain(input_image, target_class)

plt.figure(figsize=(6, 6))
plt.imshow(input_image.cpu().permute(1, 2, 0))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.colorbar()
plt.title(f"Class: {target_class}")
plt.show()
