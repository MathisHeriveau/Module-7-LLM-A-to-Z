import torch
import torch.nn as nn
from torch.nn import functional as F

# ****************************************************** #
#               ForwardLayer Class                       #
# ****************************************************** #
class ForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_size, 6*embed_size, bias=BIAS),
            nn.GELU(),
            nn.Linear(embed_size*6, embed_size, bias=BIAS),
            nn.Dropout(dropout),
        )
        
    def forward(self,x):
        return self.network(x)
    
    