from math import log2, ceil
import torch
from SOT import SOT


class Conv2dSOT(SOT):

    def __init__(self, number_of_leaves: int, kernel_size: int, stride: int = 2, padding: int = 0, lr: float = 0.3, device = torch.device("cpu")):
        super().__init__(number_of_leaves, kernel_size * kernel_size, lr, device)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def img2patches(self, x):
        padded = torch.nn.functional.pad(x, [self.padding] * 4, "constant", 0)
        p = padded.unfold(0, self.kernel_size, self.stride).unfold(1, self.kernel_size, self.stride)
        out = p.reshape(p.shape[0] * p.shape[1], p.shape[2] * p.shape[3])
        return out

    def forward(self, X):
        X = self.img2patches(X)
        indices, bmu_indices, bmu_dists = self._propagate_through_tree(X, patch_number = X.shape[0])
        neighborhood_lrs = torch.gather(input=self.learning_rates, index= indices.flatten(), dim=0).reshape(indices.shape)
        neighborhood_updates = (neighborhood_lrs.unsqueeze(2).to(self.device) * (X.unsqueeze(1).to(self.device) - self.nodes[1:,:].unsqueeze(0).to(self.device))).mean(0)
        self.nodes[1:, :] += neighborhood_updates
        return bmu_indices
