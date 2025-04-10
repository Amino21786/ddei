import torch
import torch.nn as nn



#Version from ddei version 1.0 made
from ..utils.fft import *
from .datalayer import DCLayer



class DCCNN(nn.Module):
    """
    Deep Cascade of CNNs (DCCNN) architecture for iterative image reconstruction.

    Alternates between a CNN-based regularizer and data consistency (DC) step.

    :param int num_cascades: Number of cascaded iterations (CNN + DC)
    :param int chans: Number of channels in intermediate CNN layers
    :param nn.Module datalayer: Data consistency layer used in each cascade
    """

    def __init__(
            self,
            num_cascades: int = 10,
            chans: int = 64,
            datalayer=DCLayer(),
    ):
        super().__init__()

        self.num_cascades = num_cascades
        self.chans = chans

        self.cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, chans, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chans, chans, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chans, 2, kernel_size=3, padding=1),
            ) for _ in range(num_cascades)
        ])

        self.datalayers = nn.ModuleList([datalayer for _ in range(num_cascades)])

    def forward(self, x, y, mask):
        """
        Args:
            x: Zero-filled image, shape (B, W, H, T, C)
            y: Input k-space, shape (B, W, H, T, C) (for DC)
            mask: Undersampling mask, shape (B, W, H, C, T)
        Returns:
            x: Reconstructed image, shape (B, W, H, T, C)
        """
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, W, H, T
        y = y.permute(0, 4, 1, 2, 3).contiguous()  # B, C, W, H, T

        B, C, W, H, T = x.shape
        x = x.view(B * T, C, W, H)
        y = y.view(B, C, W, H, T)

        for i in range(self.num_cascades):
            x = self.cnn_blocks[i](x)  # B*T, 2, W, H
            x = x.view(B, T, C, W, H).permute(0, 3, 4, 1, 2)  # B, W, H, T, C
            x = self.datalayers[i](x, y.permute(0, 2, 3, 4, 1), mask)
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(B * T, C, W, H)

        x = x.view(B, C, T, W, H).permute(0, 3, 4, 2, 1)  # B, W, H, T, C
        return x


class ArtifactRemovalDCCNN(nn.Module):
    r"""
    Artifact removal architecture :math:`\phi(A^{\top}y)` using DCCNN.

    Performs pseudo-inverse to get zero-filled image `x_u`, then passes `x_u`, `y`,
    and the physics mask to the DCCNN network for artifact reduction.

    Also permutes dimensions so DCCNN receives the expected input shape.

    :param torch.nn.Module backbone_net: Base DCCNN network :math:`\phi`, can be pretrained or not.
    """
    def __init__(self, backbone_net: DCCNN):
        super().__init__()
        self.backbone_net = backbone_net

    def forward(self, y: torch.Tensor, physics, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y.

        :param torch.Tensor y: Measurements of shape (B, C, T, H, W)
        :param deepinv.physics.Physics physics: Forward operator (defines A and Aᵗ)
        :return: Reconstructed image of shape (B, C, T, H, W)
        :rtype: torch.Tensor
        """
        if isinstance(physics, nn.DataParallel):
            physics = physics.module

        x_init = physics.A_adjoint(y)  # B,C,T,H,W
        mask = physics.mask.float()    # B,C,T,H,W

        x_init = x_init.permute(0, 4, 3, 2, 1)  # -> B,W,H,T,C
        y = y.permute(0, 4, 3, 2, 1)
        mask = mask.permute(0, 4, 3, 2, 1)

        x_hat = self.backbone_net(x_init, y, mask)  # B,W,H,T,C

        return x_hat.permute(0, 4, 3, 2, 1)  # -> B,C,T,H,W