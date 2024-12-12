import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from torchvision import models, transforms
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


test_dataset = PneumoniaMNIST(split="test", download=True, transform=data_transform, size=224)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


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

model.load_state_dict(torch.load("best_vgg16_multiclass.pth"))
model.to(device)
model.eval()


target_layer = model.features[28]
grad_cam = LayerGradCam(model, target_layer)


for i, (images, labels) in enumerate(test_loader):
    if i == 2:
        images, labels = images.to(device), labels.to(device)
        outputs = torch.sigmoid(model(images))
        predicted = torch.argmax(outputs, dim=1).item()


        attributions = grad_cam.attribute(images, target=predicted)


        attributions_resized = F.interpolate(
            attributions, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False
        )[0].detach().cpu().numpy()


        attributions_resized = (attributions_resized - attributions_resized.min()) / (
                attributions_resized.max() - attributions_resized.min())


        image_np = images[0, 0].cpu().numpy()


        heatmap = plt.cm.inferno(attributions_resized[0])[:, :, :3]
        overlay = 0.5 * heatmap + 0.5 * np.stack([image_np] * 3, axis=-1)


        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_np, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title(f"Input Image\nPrediction: {predicted}\nLabel: {labels.item()}")

        axes[1].imshow(overlay)
        axes[1].axis("off")
        axes[1].set_title("Grad-CAM Overlay")

        plt.tight_layout()
        plt.show()


        plt.imsave(f"gradcam_overlay_sample_{i}.png", overlay)
        break
