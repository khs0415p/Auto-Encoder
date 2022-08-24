import torch.nn as nn

class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout
        # self.height = config.height
        # self.width = config.width

        # mnist (28x28)
        self.encoder = nn.Sequential(
            nn.Linear(784, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        latent_variable = self.encoder(x)
        output = self.decoder(latent_variable)
        output = output.view(batch_size, -1, 28, 28)

        return output, latent_variable


class ConvAE(nn.Module):
    def __init__(self) -> None:
        super(ConvAE, self).__init__()