import torch
import torch.nn as nn
import torch.nn.functional as F
from scimilarity.nn_models import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderWithLoRA(nn.Module):
    def __init__(self, base_encoder, lora_rank=16):
        super().__init__()
        self.base_encoder = base_encoder
        self.lora_rank = lora_rank
        self.lora_params = nn.ParameterDict()

        for idx, module in enumerate(base_encoder.network):
            if isinstance(module, nn.Sequential):
                # LoRA for Linear layers inside Sequential
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Linear):
                        key_prefix = f"{idx}_{name}".replace(".", "_")
                        A = nn.Parameter(torch.randn(lora_rank, sub_module.in_features) * 0.01)
                        B = nn.Parameter(torch.randn(sub_module.out_features, lora_rank) * 0.01)
                        self.lora_params[f"{key_prefix}_A"] = A
                        self.lora_params[f"{key_prefix}_B"] = B
            elif isinstance(module, nn.Linear):
                # top-level Linear layer
                key_prefix = f"{idx}"
                A = nn.Parameter(torch.randn(lora_rank, module.in_features) * 0.01)
                B = nn.Parameter(torch.randn(module.out_features, lora_rank) * 0.01)
                self.lora_params[f"{key_prefix}_A"] = A
                self.lora_params[f"{key_prefix}_B"] = B

    def forward(self, x):
        out = x
        for idx, module in enumerate(self.base_encoder.network):
            # Check if module is Sequential
            if isinstance(module, nn.Sequential):
                for name, sub_module in module.named_children():
                    if isinstance(sub_module, nn.Linear):
                        key_prefix = f"{idx}_{name}".replace(".", "_")
                        A = self.lora_params[f"{key_prefix}_A"]
                        B = self.lora_params[f"{key_prefix}_B"]
                        W_eff = sub_module.weight + B @ A
                        b = sub_module.bias
                        out = F.linear(out, W_eff, b)
                    else:
                        out = sub_module(out)
            elif isinstance(module, nn.Linear):
                # top-level Linear layer (like network[2])
                key_prefix = f"{idx}".replace(".", "_")
                A = self.lora_params[f"{key_prefix}_A"]
                B = self.lora_params[f"{key_prefix}_B"]
                W_eff = module.weight + B @ A
                b = module.bias
                out = F.linear(out, W_eff, b)
            else:
                out = module(out)
        return out



def adapt_first_layer(model, ckpt_state_dict, new_input_dim, hidden_dim):
    """
    Replace first Linear layer to accommodate new_input_dim genes.
    Preserve existing weights for overlapping genes.
    """
    first_layer_name = "network.0.1"  # adjust based on checkpoint naming
    ckpt_w = ckpt_state_dict[first_layer_name + ".weight"]  # [hidden_dim, old_input_dim]
    ckpt_b = ckpt_state_dict[first_layer_name + ".bias"]    # [hidden_dim]

    old_input_dim = ckpt_w.shape[1]
    if new_input_dim < old_input_dim:
        # Truncate weights
        new_w = ckpt_w[:, :new_input_dim].clone()
    else:
        # Keep existing weights, randomly init extra columns
        extra_cols = new_input_dim - old_input_dim
        new_w_extra = torch.randn(hidden_dim, extra_cols) * 0.02
        new_w = torch.cat([ckpt_w, new_w_extra], dim=1)

    # Replace weight and bias
    state_dict = model.state_dict()
    state_dict[first_layer_name + ".weight"] = new_w
    state_dict[first_layer_name + ".bias"] = ckpt_b
    model.load_state_dict(state_dict, strict=False)
    return model

def load_encoder(n_genes, hidden_dim):
    # --- Load encoder ---
    encoder_ckpt = torch.load("weights/encoder.ckpt", map_location=device)
    encoder = Encoder(n_genes=n_genes)
    encoder = adapt_first_layer(encoder, encoder_ckpt['state_dict'], new_input_dim=n_genes, hidden_dim=hidden_dim)
    encoder.eval()
    encoder.to(device)
    return encoder

def load_lora_encoder(n_genes, hidden_dim):
    encoder = load_encoder(n_genes, hidden_dim)
    encoder_lora = EncoderWithLoRA(encoder, lora_rank=16)

    # Freeze base encoder
    for param in encoder_lora.base_encoder.parameters():
        param.requires_grad = False

    encoder_lora.to(device)
    return encoder_lora

def load_decoder(n_genes):
    # --- Load decoder ---
    decoder = Decoder(n_genes=n_genes)
    decoder_ckpt = torch.load("weights/decoder.ckpt", map_location="cpu")
    state_dict = decoder_ckpt['state_dict']

    # Remove final layer weights from state dict (only old n_genes)
    for key in list(state_dict.keys()):
        if 'network.3' in key:  # network.3 is the final linear layer
            del state_dict[key]

    # Load remaining pretrained weights
    decoder.load_state_dict(state_dict, strict=False)  # final layer remains randomly initialized
    decoder.eval()
    decoder.to(device)
    return decoder


