# -*- coding: utf-8 -*-
import torch

from ..utils import lerp


class WGAN_ACGAN(torch.nn.Module):
    """WGAN + AC-GAN Loss Function

    Used as a loss function for training generators in GANs

    Note:
        References:
        `WGAN <https://arxiv.org/pdf/1704.00028.pdf>`_,
        `AC-GAN <https://arxiv.org/pdf/1610.09585.pdf>`_
    """

    def __init__(self, cond_weight: float = 1.0):
        super().__init__()
        self.cond_weight = cond_weight

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return -x.mean()


class WGANGP_ACGAN(torch.nn.Module):
    """WGAN-GP + AC-GAN Loss Function

    Used as a loss function for training discriminators in GANs

    Note:
        References:
        `WGAN-GP <https://arxiv.org/pdf/1704.00028.pdf>`_,
        `AC-GAN <https://arxiv.org/pdf/1610.09585.pdf>`_
    """

    def __init__(
        self, generator, discriminator, drift: float = 0.001, use_gp: bool = False
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = next(discriminator.parameters()).device

        self.drift = drift
        self.use_gp = use_gp

    def forward(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        alpha=0,
        reg_lambda=10,
    ):
        real_outputs = self.discriminator(real_images, alpha)
        fake_outputs = self.discriminator(fake_images, alpha)

        loss = fake_outputs.mean() - real_outputs.mean()

        if self.drift != 0:
            loss += self.drift * real_outputs.pow(2).mean()

        if self.use_gp:
            minibatch_size = real_images.shape[0]
            mixing_factors = torch.rand((minibatch_size, 1, 1, 1), device=self.device)

            mixed_images = lerp(real_images, fake_images, mixing_factors)
            mixed_images.requires_grad_(True)

            mixed_outputs = self.discriminator(mixed_images, alpha)

            gradient = torch.autograd.grad(
                outputs=mixed_outputs,
                inputs=mixed_images,
                grad_outputs=torch.ones_like(mixed_outputs),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient = gradient.view(gradient.shape[0], -1)
            gradient_penalty = (
                reg_lambda * (gradient.norm(p=2, dim=1) - 1).pow(2).mean()
            )

            loss += gradient_penalty

        return loss
