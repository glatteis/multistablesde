import torch
import torchsde

class StochasticEnergyBalance(object):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    albedo_0 = 0.425
    albedo_var = 0.4
    solarconstant = 1363.0
    radiation = 0.6 * 5.67e-8
    noise_var = 0.08

    def albedo(self, u):
        return self.albedo_0 - (self.albedo_var / 2.0) * torch.tanh(u - 273.0)

    def energy_in(self, u):
        return (1.0 - self.albedo(u)) * (self.solarconstant / 4.0)

    def energy_out(self, u):
        return self.radiation * u**4

    def f(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = self.energy_in(x[0]) - self.energy_out(x[0])
        return torch.cat([f], dim=1)

    def g(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = x[0] * self.noise_var
        return torch.cat([f], dim=1)

    @torch.no_grad()
    def sample(self, batch_size, ts, noise_std, normalize):
        x0 = (torch.randn(batch_size, 1) * 20.0) + 270.0
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std)
        return xs
