import torch
import torchvision

# The resnet model to getting the image features and insert it to the transformer model
class EncoderCNNtrain18(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super(EncoderCNNtrain18, self).__init__()
        self.embedding_dim = embedding_dim
        resnet = torchvision.models.resnet18()

        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)
        self.dense = torch.nn.Linear(512, embedding_dim)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,512,8,8)
        features = features.permute(0, 2, 3, 1)  # (batch_size,8,8,512)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,64,512)
        features = self.dense(features)
        return features
