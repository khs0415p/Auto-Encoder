import torch.nn as nn
import torch

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
    def __init__(self, config, channel=1) -> None:
        super(ConvAE, self).__init__()

        self.config = config
        self.channel = channel
        self.conv_dims = self.config.conv_dims
        self.conv_dims = [channel] + self.conv_dims
        self.latent_dim = self.config.latent_dim
        self.kernel_sizes = self.config.kernel_sizes
        self.step_pool = (len(self.conv_dims)-1) // 2

        # encoder
        enc_layers = []
        img_len = 28
        for i in range(len(self.conv_dims)-1):
            enc_layers.append(nn.Conv2d(in_channels=self.conv_dims[i], out_channels=self.conv_dims[i+1], kernel_size=self.kernel_sizes[i], stride=1, padding=1))
            enc_layers.append(nn.ReLU())

            if (i != 0 and i % self.step_pool == 0) or i == (len(self.conv_dims)-2):
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                img_len = img_len // 2

        # add fc
        self.encoders = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(in_features=img_len**2 * self.conv_dims[-1], out_features=img_len**2 * self.conv_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(in_features=img_len**2 * self.conv_dims[-1] // 2, out_features=self.latent_dim),
            nn.ReLU()
        )

        # decoder
        dec_layers = []
        for i in reversed(range(1, len(self.conv_dims))):
            if i == 1 or (i != len(self.conv_dims)-1 and (i+1) % self.step_pool == 0):
                dec_layers.append(nn.ConvTranspose2d(in_channels=self.conv_dims[i], out_channels=self.conv_dims[i-1], kernel_size=2, stride=2, padding=0))

            else:
                dec_layers.append(nn.ConvTranspose2d(in_channels=self.conv_dims[i], out_channels=self.conv_dims[i-1], kernel_size=3, stride=1, padding=1))

            dec_layers.append(nn.ReLU())

        dec_layers.pop()

        self.decoders = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=img_len**2 * self.conv_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(in_features=img_len**2 * self.conv_dims[-1] // 2, out_features= img_len**2 * self.conv_dims[-1]),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(self.conv_dims[-1], img_len, img_len)),
            *dec_layers,
            nn.Sigmoid()
        )

    def forward(self, x):

        latent_variable = self.encoders(x)
        output = self.decoders(latent_variable)

        return output, latent_variable