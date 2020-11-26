from math import log2, ceil
import torch


class SOTLayer(torch.nn.Module):

    def __init__(self, number_of_leaves: int, input_dim: int, lr: float = 0.3, device = torch.device("cpu")):
        super(SOTLayer, self).__init__()
        self.leaf_num = 2**ceil(log2(number_of_leaves))
        self.depth = log2(self.leaf_num)
        self.learning_rates = self.learning_rates_per_branch(lr)
        self.nodes = torch.nn.Parameter(data= torch.Tensor((self.leaf_num*2)-1, input_dim), requires_grad=False).to(device)
        self.nodes.data.uniform_(0, 1)
        self.node_indices = torch.arange(self.nodes.shape[0])
        self.device = device

    def get_leaves(self):
        return self.nodes[int((self.nodes.shape[0]-1) / 2) : ]

    def pnorm(self, x1, x2, p=2):
        return torch.pow(torch.pow(x1 - x2.unsqueeze(dim=0).unsqueeze(dim=0), p).sum(dim=2), p)

    def learning_rates_per_branch(self, lr: float):
        return (lr * 2 ** torch.arange(1,self.depth+2, dtype=torch.float)) / (2**(self.depth+1))

    def _propagate_through_tree(self, X):
        patch_num = 1 #X.shape[2]
        start, num = 1, 2
        layers = []
        layer_state = torch.zeros(patch_num, 1, dtype=int)
        update_amount_indices = torch.zeros(2, dtype=int)
        for n in range(1, int(log2(self.leaf_num)+1)):
            nodes_per_layer = num ** n
            layer = torch.arange(start, + start + nodes_per_layer)
            layer_state = layer_state.repeat_interleave(2).reshape(patch_num,nodes_per_layer)
            max_val = layer_state.max(dim=1).values.unsqueeze(dim=1)
            competing_indices = layer.repeat(patch_num,1)[layer_state == max_val].reshape(patch_num, num)
            competing_nodes = self.nodes[competing_indices].clone().to('cpu')
            dist = self.pnorm(competing_nodes, X)
            bmu_dists, bmu = torch.min(dist, 1)
            bmu_index = competing_indices.squeeze()[bmu.squeeze()]
            layer_state = layer_state.add((layer == bmu_index).to(torch.int64))
            layers.append(layer_state)
            start += nodes_per_layer
        return torch.cat(layers, dim=1), bmu_index, bmu_dists

    def forward(self, X):
        indices, bmu_index, bmu_dists = self._propagate_through_tree(X)
        neighborhood_lrs = torch.gather(input=self.learning_rates, index= indices.flatten(), dim=0).reshape(indices.shape)
        neighborhood_updates = (neighborhood_lrs.squeeze().unsqueeze(1).to(self.device) * (X.unsqueeze(0).to(self.device) - self.nodes[1:,:].unsqueeze(0).to(self.device))).mean(0)
        self.nodes[1:, :] += neighborhood_updates
        return bmu_index
    
