import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from medmnist import INFO
from medmnist.dataset import PathMNIST
from medmnist import PneumoniaMNIST
from torchvision import models, transforms

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
num_classes = 1


transform = transforms.Compose([
    transforms.ToTensor()
])


test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True,size=224)


class RISE:
    def __init__(self, model, input_size, num_masks=5000, mask_size=8):

        self.model = model
        self.input_size = input_size
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.masks = self._generate_masks()

    def _generate_masks(self):

        np.random.seed(42)
        masks = np.random.choice([0, 1], size=(self.num_masks, self.mask_size, self.mask_size), p=[0.5, 0.5])
        masks = masks.astype(np.float32)
        masks = np.array([
            cv2.resize(mask, self.input_size, interpolation=cv2.INTER_LINEAR) for mask in masks
        ])
        return torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # (num_masks, 1, H, W)

    def explain(self, image, target_class):

        image = image.unsqueeze(0)
        batch_size = 32
        heatmap = torch.zeros(self.input_size, dtype=torch.float32)


        for i in tqdm(range(0, self.num_masks, batch_size)):
            batch_masks = self.masks[i:i + batch_size]
            masked_images = image * batch_masks.cuda()
            outputs = self.model(masked_images).detach().cpu()  
            scores = outputs.squeeze(1)


            for j, mask in enumerate(batch_masks):
                heatmap += mask.squeeze(0) * scores[j]


        heatmap /= self.num_masks
        return heatmap.numpy()


image, label = test_dataset[3]
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import tqdm as notebook_tqdm

class UNetForClassification(nn.Module):
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet",
                 in_channels=1, num_classes=1, dropout_rate=0.5, freeze_encoder=False):
        super(UNetForClassification, self).__init__()

        # Initialize UNet encoder
        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1  # Dummy classes for segmentation; we're only using the encoder
        )
        self.encoder = unet.encoder

        # Optionally freeze the encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Add global average pooling, dropout, and classification head
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_classes)  # Fully connected layer
        self.norm = nn.BatchNorm1d(num_classes)  # Normalization layer
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else None  # Sigmoid for binary classification

    def forward(self, x):
        # Extract features from the encoder
        features = self.encoder(x)[-1]  # Deepest feature map

        # Apply global average pooling
        pooled = self.pooling(features).view(features.size(0), -1)  # Flatten
        pooled = self.dropout(pooled)

        # Apply the fully connected layer and normalization
        output = self.fc(pooled)
        output = self.norm(output)

        # Activation (Sigmoid for binary classification)
        if self.sigmoid:
            output = self.sigmoid(output)

        return output
# Instantiate the model
if __name__ == "__main__":
    model = UNetForClassification(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        num_classes=1,  # Binary classification
        dropout_rate=0.4,
        freeze_encoder=False  # Allow the encoder to train
    )
criterion = nn.BCELoss()

model = UNetForClassification(num_classes=1)  # Binary classification typically uses 1 output channel

# Load the model weights
state_dict = torch.load("best_unet_restnet18_binaryclass.pth", weights_only=True, map_location=torch.device('cuda'))
model.load_state_dict(state_dict)
# Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model = model.to(device)





input_image = image.to(device)


rise = RISE(model, input_size=(224, 224), num_masks=20000, mask_size=8)
target_class = label.item()
heatmap = rise.explain(input_image, target_class)

plt.figure(figsize=(6, 6))
plt.imshow(input_image.cpu().permute(1, 2, 0))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.colorbar()
plt.title(f"Class: {target_class}")
plt.show()
