import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import optuna
from simpleGAN import Discriminator, Generator

warnings.filterwarnings("ignore")


def objective(trial):
    device = torch.device("mps")
    print(f"Device: {device}")
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    img_dim = 28 * 28 * 1
    z_dim = 64
    num_epochs = trial.suggest_int("num_epochs", 50, 150, step=10)

    disc = Discriminator(img_dim).to(device)
    gen = Generator(z_dim, img_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    writer = SummaryWriter(f"runs/GAN_MNIST_trial_{trial.number}")
    writer.add_hparams(trial.params, {})

    step = 0
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (real, _) in enumerate(train_loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    writer.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                    writer.add_image("MNIST Real Images", img_grid_real, global_step=step)
                    writer.add_scalar("Loss D", lossD, global_step=step)
                    writer.add_scalar("Loss G", lossG, global_step=step)
                    step += 1

    # Validation
    with torch.no_grad():
        val_lossD, val_lossG = 0, 0
        for batch_idx, (real, _) in enumerate(val_loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            val_lossD += lossD.item()

            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            val_lossG += lossG.item()

        val_lossD /= len(val_loader)
        val_lossG /= len(val_loader)

    writer.close()

    return val_lossD + val_lossG


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print(f"Best trial: {study.best_trial.params}")
