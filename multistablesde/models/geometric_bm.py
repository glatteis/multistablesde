import torch
import torchsde


class GeometricBM(object):
    noise_type = "diagonal"
    sde_type = "ito"

    mu = 1.0
    sigma = 0.5

    state_space_size = 1

    def f(self, t, y):
        return y * self.mu

    def g(self, t, y):
        return y * self.sigma

    @torch.no_grad()
    def sample(self, batch_size, ts, normalize, device, bm=None):
        x0 = (torch.randn(batch_size, 1) * 0.03**2 + 0.1).to(device)
        xs = torchsde.sdeint(self, x0, ts, bm=bm)
        if normalize:
            mean, std = torch.mean(xs[0, :, :], dim=(0, 1)), torch.std(
                xs[0, :, :], dim=(0, 1)
            )
            xs.sub_(mean).div_(std)
            return xs, mean, std
        else:
            return xs
