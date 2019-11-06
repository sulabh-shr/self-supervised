import torch.nn as nn
from torchvision import models


class MetricLearningNet(nn.Module):
    def __init__(self):
        super(MetricLearningNet, self).__init__()
        self.embedder = models.resnet50(pretrained=False)

    def forward(self, ref, pos, neg):
        ref_emb = self.embedder(ref)
        ref_pos = self.embedder(pos)
        ref_neg = self.embedder(neg)

        return ref_emb, ref_pos, ref_neg

