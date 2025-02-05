import torch.nn as nn

from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016)
    to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the
    average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j, ns=None):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        if ns is not None:
            h_ns = self.encoder(ns)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        if ns is not None:
            z_ns = self.projector(h_ns)
            return h_i, h_j, h_ns, z_i, z_j, z_ns

        return h_i, h_j, z_i, z_j

