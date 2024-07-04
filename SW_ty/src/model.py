from torch import nn
import torchvision.models as models

class EfficientNetB7Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=2560):
        super(EfficientNetB7Classifier, self).__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet_b7(x)
        x = self.sigmoid(x)
        return x