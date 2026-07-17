import torch

class GatedContextPredictor(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim
        # Causal 1D convolution over sequence length to capture local token transitions
        self.conv = torch.nn.Conv1d(
            in_channels=emb_dim,
            out_channels=hidden_dim * 2, # Gated output split in half for GLU
            kernel_size=3,
            padding=2, # Causal padding (we will slice off the right padding)
        )
        self.glu = torch.nn.GLU(dim=1)
        self.proj = torch.nn.Linear(hidden_dim, emb_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [B, U, D]
        B, U, D = x.shape
        x_in = x.transpose(1, 2) # [B, D, U]
        
        # Apply causal convolution
        conv_out = self.conv(x_in) # [B, 2*D, U + 2]
        conv_out = conv_out[..., :U] # Slice right padding to preserve causality: [B, 2*D, U]
        
        # Apply Gated Linear Unit
        gated = self.glu(conv_out) # [B, D, U]
        gated = gated.transpose(1, 2) # [B, U, D]
        
        # Project back, residual, and normalize
        out = self.proj(gated)
        out = self.norm(x + self.dropout(out))
        return out
