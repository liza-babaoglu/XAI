import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, visualization
from torchvision import models, transforms
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
# 数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
])

# 加载数据
test_dataset = PneumoniaMNIST(split="test", download=True, transform=data_transform, size=224)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
model.features[0] = torch.nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
num_classes = 2
model.classifier[5] = torch.nn.Dropout(0.5)
model.classifier[6] = torch.nn.Linear(4096, num_classes)
# layer_idx=24  # 第 24 层索引
# conv_layer = model.features[layer_idx]


# model.features[layer_idx] = nn.Conv2d(
#     in_channels=conv_layer.in_channels,  # 保持输入通道一致
#     out_channels=conv_layer.out_channels,  # 保持输出通道一致
#     kernel_size=5,  # 改为 5x5 卷积
#     stride=1,       # 保持步幅
#     padding=2       # 为保持输出尺寸不变，需设置 padding=(kernel_size // 2)
# )
model.load_state_dict(torch.load("best_vgg16_multiclass.pth"))
model.to(device)
model.eval()

# 初始化 Grad-CAM
layer_names = [name for name, module in model.features.named_children() if isinstance(module, torch.nn.Conv2d)]
layer_cams = {name: LayerGradCam(model, model.features[int(name)]) for name in layer_names}

# 遍历数据并生成 Grad-CAM 可视化
target_indices = {2}  # 示例索引
processed_count = 0
max_samples = len(target_indices)

for i, (images, labels) in enumerate(test_loader):
    if processed_count >= max_samples:
        break
    images, labels = images.to(device), labels.to(device)
    outputs = torch.sigmoid(model(images))
    predicted = torch.argmax(outputs, dim=1).item()

    if i not in target_indices:
        continue
    print(labels)
    # 遍历每个卷积层
    for layer_name, grad_cam in layer_cams.items():
        # 计算 Grad-CAM
        attributions = grad_cam.attribute(images, target=predicted)

        # 转换为可视化格式
        attribution_np = attributions[0].detach().cpu().numpy()
        attribution_map = np.mean(attribution_np, axis=0)  # 平均通道维度

        # 原始输入图像
        image_np = images[0, 0].cpu().numpy()

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_np, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title(f"Input Image\nPrediction: {predicted}\nLayer: {layer_name}")

        visualization.visualize_image_attr(
            np.expand_dims(attribution_map, axis=-1),
            np.expand_dims(image_np, axis=-1),
            method="heat_map",
            cmap="inferno",
            show_colorbar=False,
            plt_fig_axis=(fig, axes[1]),
            use_pyplot=False
        )

        plt.tight_layout()
        plt.show()

    processed_count += 1
    if processed_count >= max_samples:
        break
