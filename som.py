import torch


class SOM(torch.nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, N, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.N = N
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = N / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(N, dim)
        self.locations = torch.arange(N)#.unsqueeze(0)
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: np.linalg.norm(vect-self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):
        dists = self.pdist(x.unsqueeze(0), self.weights)
        _, bmu_index = torch.min(dists, 0)

        learning_rate_op = 1.0 - it/self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op
        bmu_distance_squares = torch.pow((self.locations - bmu_index).float(), 2)
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))

        learning_rate_op = alpha_op * neighbourhood_func
        delta = learning_rate_op.unsqueeze(1) * (x.unsqueeze(0) - self.weights)
        self.weights += delta
