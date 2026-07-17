import torch

class FSMNContextPredictor(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, memory_depth=4, num_layers=2):
        super().__init__()
        self.memory_depth = memory_depth
        self.num_layers = num_layers
        
        # We stack multiple feedforward + causal memory block layers
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.ModuleDict({
                "proj_in": torch.nn.Linear(emb_dim, hidden_dim),
                # Learnable FIR filter memory coefficients
                "memory_weights": torch.nn.Parameter(torch.randn(memory_depth, hidden_dim) * 0.02),
                "proj_out": torch.nn.Linear(hidden_dim, emb_dim),
                "norm": torch.nn.LayerNorm(emb_dim),
                "activation": torch.nn.GELU()
            })
            self.layers.append(layer)
            
    def forward(self, x):
        # x shape: [B, U, D] (concatenated embeddings)
        B, U, D = x.shape
        
        current_x = x
        for layer in self.layers:
            # Feedforward projection
            h = layer["activation"](layer["proj_in"](current_x)) # [B, U, H]
            
            # Compute Causal Tapped-Delay Memory Block
            # For each lookback index i, shift the hidden states to the right (causal shift)
            memory_sum = torch.zeros_like(h)
            for i in range(1, self.memory_depth + 1):
                # Shift elements by i to the right
                shifted_h = torch.zeros_like(h)
                if U > i:
                    shifted_h[:, i:, :] = h[:, :-i, :]
                
                # Apply channel-wise learnable memory weights
                w = layer["memory_weights"][i - 1].unsqueeze(0).unsqueeze(0) # [1, 1, H]
                memory_sum += shifted_h * w
                
            # Combine current projection with memory block
            h_tilde = h + memory_sum
            
            # Project back and apply residual norm
            out = layer["proj_out"](h_tilde)
            current_x = layer["norm"](current_x + out)
            
        return current_x
