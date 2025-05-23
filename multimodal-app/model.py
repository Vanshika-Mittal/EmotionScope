import torch
import torch.nn.functional as F
from torchvision import models

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, tfidf_dim, num_classes):
        super(MultimodalClassifier, self).__init__()

        # Image branch
        self.img_model = models.resnet18(pretrained=False)
        self.img_model.fc = torch.nn.Identity()
        self.img_proj = torch.nn.Linear(512, 512)

        # Text branch
        self.tfidf_proj = torch.nn.Linear(tfidf_dim, 512)

        # Fusion and classification
        self.dropout = torch.nn.Dropout(0.3)
        self.fusion = torch.nn.Linear(1024, 512)
        self.classifier = torch.nn.Linear(512, num_classes)

    def forward(self, tfidf_input, image_input):
        img_feat = self.img_model(image_input)
        img_feat = self.img_proj(img_feat)
        tfidf_feat = self.tfidf_proj(tfidf_input)

        # Normalize features
        img_feat = F.normalize(img_feat, p=2, dim=1)
        tfidf_feat = F.normalize(tfidf_feat, p=2, dim=1)

        combined = torch.cat((img_feat, tfidf_feat), dim=1)
        combined = self.dropout(combined)
        fused = F.relu(self.fusion(combined))
        output = self.classifier(fused)
        return output
