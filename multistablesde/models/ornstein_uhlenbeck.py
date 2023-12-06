import torch
import torchsde


class OrnsteinUhlenbeck(object):
    noise_type = "diagonal"
    sde_type = "ito"

    mu = 0.0
    sigma = 1
    psi = 1

    state_space_size = 1

    def f(self, t, u):
        return self.mu * t - self.sigma * u

    def g(self, t, u):
        return torch.full_like(u, self.psi)

    @torch.no_grad()
    def sample(self, batch_size, ts, normalize, device, bm=None):
        x0 = (torch.randn(batch_size, 1) * 0.1 + 0.5).to(device)
        xs = torchsde.sdeint(self, x0, ts, bm=bm)
        mean, std = torch.mean(xs[0, :, :], dim=(0, 1)), torch.std(
            xs[0, :, :], dim=(0, 1)
        )
        if normalize:
            xs.sub_(mean).div_(std)
        return xs, mean, std
