import torch.nn as nn

class MaximizeMSELoss(nn.Module):
    def __init__(self):
        super(MaximizeMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        return -self.mse_loss(output, target)