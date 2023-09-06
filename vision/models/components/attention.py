from __future__ import annotations

import torch
import torch.nn as nn


class MHSA(nn.Module):
    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
