"""
MultiScaleCNN1D: lightweight multi-branch 1D-CNN for SGCC time-series.

Input:  [N, C, T]  (SGCC: 5 channels x 1035 time steps)
Output: [N]         sigmoid logits for binary electricity-theft classification

Four parallel branches with kernel sizes {3, 5, 7, 11} capture local patterns
at different temporal scales. Each branch uses weight-normalised 1D conv,
BatchNorm, GELU and adaptive average pooling. The pooled features are
concatenated and fed to a compact classification head.
"""
import math
import torch
import torch.nn as nn


class MultiScaleBranch(nn.Module):
    """Single 1D-CNN branch with a fixed kernel size."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False)
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: [N, C, T]
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        # global temporal average pooling -> [N, out_channels, 1]
        out = nn.functional.adaptive_avg_pool1d(out, 1)
        return out.squeeze(-1)


class MultiScaleCNN1D(nn.Module):
    """Multi-scale 1D CNN classifier.

    Parameters
    ----------
    in_channels : int
        Number of input channels (SGCC: 5).
    kernel_sizes : tuple of int
        Kernel sizes for the parallel branches.
    branch_channels : int
        Output channels of each branch.
    hidden_dim : int
        Hidden dimension of the classification head.
    dropout : float
        Dropout probability applied after each branch and in the head.
    """

    def __init__(self, in_channels=5, kernel_sizes=(3, 5, 7, 11),
                 branch_channels=16, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.branch_channels = branch_channels

        self.branches = nn.ModuleList([
            MultiScaleBranch(in_channels, branch_channels, k, dropout=dropout)
            for k in kernel_sizes
        ])

        concat_dim = len(kernel_sizes) * branch_channels

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, C, T].

        Returns
        -------
        torch.Tensor
            Logits of shape [N].
        """
        if x.dim() == 2:
            # [N, T] -> [N, 1, T]
            x = x.unsqueeze(1)

        # Handle NaN/Inf safely by zero-filling (input should already be clean)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        branch_feats = [branch(x) for branch in self.branches]
        fused = torch.cat(branch_feats, dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick smoke test
    model = MultiScaleCNN1D(in_channels=5)
    x = torch.randn(8, 5, 1035)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
