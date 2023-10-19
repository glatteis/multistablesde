import torch
import torchsde
import math


class FitzHughNagumoKeno(object):
    noise_type = "diagonal"
    sde_type = "ito"

    alpha = 2.0
    t_x = 1
    t_y = 0.1
    y = 0.0
    sigmax = 0.1
    sigmay = 0.1

    state_space_size = 2

    # drift(u, fhn::FitzHughNagumoModel, t) = [driftx(u, fhn, t), drifty(u, fhn, t)]
    # driftx(u, fhn::FitzHughNagumoModel, t) = fhn.ɑ₁ * u[1] - fhn.ɑ₃ * u[1]^3 + fhn.b * u[2]
    # drifty(u, fhn::FitzHughNagumoModel, t) = tan(fhn.β) * u[2] - u[1] + fhn.c
    # diffusion(u, fhn::FitzHughNagumoModel, t) = [fhn.σx, fhn.σy]

    def driftx(self, u, t):
        return  (1 / self.t_y) * (self.alpha * (u[0] - u[0]**3) - u[1])

    def drifty(self, u, t):
        return (1 / self.t_x) * (u[0] - self.y)

    def diffusionx(self, u, t):
        return torch.full_like(u, self.sigmay)

    def diffusiony(self, u, t):
        return torch.full_like(u, self.sigmax)

    def f(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = [self.driftx(x, t), self.drifty(x, t)]
        return torch.cat(f, dim=1)

    def g(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = [self.diffusionx(x[0], t), self.diffusiony(x[1], t)]
        return torch.cat(f, dim=1)

    @torch.no_grad()
    def sample(self, batch_size, ts, normalize, device, bm=None, project=True):
        x0 = (torch.randn(batch_size, 2) * 1.0).to(device)
        """Sample data for training. Store data normalization constants if necessary."""
        # Throw away second dimension
        xs_ = torchsde.sdeint(self, x0, ts, bm=bm)
        if project:
            xs = xs_[:, :, 0:1]
        else:
            xs = xs_
        if normalize:
            mean, std = torch.mean(xs[0, :, :], dim=(0, 1)), torch.std(
                xs[0, :, :], dim=(0, 1)
            )
            xs.sub_(mean).div_(std)
            return xs, mean, std
        else:
            return xs
