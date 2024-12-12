from captum.attr import GradientShap
import torch
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST,INFO
import torch.nn as nn
from torchvision import models, transforms
from captum.attr import IntegratedGradients, visualization
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.empty_cache()
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_dataset = PneumoniaMNIST(split="test", download=True, transform=data_transform, size=224)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
num_classes = len(info['label'])

model = models.vgg16(pretrained=False)
model.features[0] = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
model.classifier[5] = nn.Dropout(0.5)
model.classifier[6] = torch.nn.Linear(4096, num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_vgg16_multiclass.pth"))

model.to(device)
gs = GradientShap(model)
model.eval()


target_indices = {27}
processed_count = 0
max_samples = len(target_indices)

for i, (images, labels) in enumerate(test_loader):
    if processed_count >= max_samples:
        break
    images, labels = images.to(device), labels.to(device)
    images.requires_grad_()


    outputs = torch.sigmoid(model(images))
    predicted = (outputs > 0.5).long()
    predicted_cpu = predicted.cpu()
    labels_cpu = labels.cpu()


    for j in range(images.size(0)):
        global_index = i * test_loader.batch_size + j

        if global_index not in target_indices:
            continue

        image = images[j].unsqueeze(0)
        label = labels_cpu[j].item()
        predicted_label = predicted_cpu[j].argmax().item()

        is_correct = "Correct" if predicted_label == label else "Incorrect"


        baseline_dist = torch.cat([torch.zeros_like(image), torch.ones_like(image) * 0.5])
        attributions = gs.attribute(
            inputs=image,
            baselines=baseline_dist,
            target=label,
            n_samples=50,
            stdevs=0.09
        )

        attributions_3d = np.expand_dims(attributions[0, 0].detach().cpu().numpy(), axis=-1)
        image_3d = np.expand_dims(image[0, 0].detach().cpu().numpy(), axis=-1)


        fig, axes = plt.subplots(1, 2, figsize=(8, 8))

        axes[0].imshow(image_3d.squeeze(), cmap="gray")
        axes[0].axis("off")
        axes[0].set_title(
            f"Index: {global_index}\nPrediction: {predicted_label}\nLabel: {label}\n{is_correct}"
        )

        visualization.visualize_image_attr(
            attributions_3d,
            image_3d,
            method="heat_map",
            cmap="coolwarm",
            show_colorbar=False,
            plt_fig_axis=(fig, axes[1]),
            use_pyplot=False
        )

        plt.tight_layout()
        plt.show()

        processed_count += 1

    if processed_count >= max_samples:
        break
