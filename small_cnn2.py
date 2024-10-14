from collections import OrderedDict
import torch.nn as nn


class SmallCNN2(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN2, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(inplace=True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(32, 64, 3)),
            ('relu2', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 5 * 5, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc2.weight, 0)
        nn.init.constant_(self.classifier.fc2.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(features.size(0), -1))
        return logits