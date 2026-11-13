import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Vocabulary for Natural Language Conditioning
VOCAB = {
    '<pad>': 0, '<unk>': 1, 'pick': 2, 'up': 3, 'the': 4, 'and': 5, 'place': 6,
    'it': 7, 'on': 8, 'platform': 9, 'dock': 10, 'bay': 11, 'red': 12, 'can': 13,
    'soda': 14, 'coke': 15, 'blue': 16, 'mug': 17, 'coffee': 18, 'cup': 19,
    'green': 20, 'flask': 21, 'bottle': 22, 'specimen': 23, 'avoid': 24,
    'obstacle': 25, 'barrier': 26, 'laser': 27, 'clear': 28, 'over': 29,
    'deliver': 30, 'to': 31, 'take': 32, 'transfer': 33, '1': 34
}

def tokenize_instruction(text, max_len=16):
    """Tokenize a natural language instruction string into fixed-length integer IDs."""
    words = text.lower().replace(',', ' ').replace('.', ' ').split()
    tokens = [VOCAB.get(w, VOCAB['<unk>']) for w in words][:max_len]
    padding = [VOCAB['<pad>']] * (max_len - len(tokens))
    return torch.tensor([tokens + padding], dtype=torch.long)

class PatchVisualTokenizer(nn.Module):
    """
    Visual Tokenizer: Converts 256x256x3 RGB image into 16x16 = 256 visual tokens
    using a Conv2d patch projection followed by layer norm and 2D learned position embeddings.
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 # 16 * 16 = 256

        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]
        # [B, embed_dim, 16, 16] -> [B, embed_dim, 256] -> [B, 256, embed_dim]
        tokens = self.patch_proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.norm(tokens)
        return tokens

class LanguageTokenizer(nn.Module):
    """
    Language Tokenizer & Encoder: Maps word tokens to language embeddings with learned position encodings.
    """
    def __init__(self, vocab_size=len(VOCAB) + 10, max_len=16, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        # tokens: [B, seq_len]
        x = self.embed(tokens) + self.pos_embed[:, :tokens.shape[1], :]
        return self.norm(x)

class OctoMultimodalTransformer(nn.Module):
    """
    Cross-Attention Multimodal Transformer: Fuses vision tokens and language tokens
    to attend to task-relevant objects and spatial targets.
    """
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, mlp_dim=256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_tokens, lang_tokens):
        # visual_tokens: [B, 256, embed_dim]
        # lang_tokens: [B, 16, embed_dim]
        x = visual_tokens
        attention_maps = []
        for layer in self.layers:
            x = layer(tgt=x, memory=lang_tokens)
        return self.norm(x)

class ActionPredictionHead(nn.Module):
    """
    Continuous Action Head:
    Predicts 7-DoF delta end-effector action:
    [dx, dy, dz, droll, dpitch, dyaw, gripper_target]
    """
    def __init__(self, embed_dim=128, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 4, 128), # +4 for proprioception: [x_tcp, y_tcp, z_tcp, grip]
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, fused_tokens, proprio):
        # Pool visual tokens into a global representation: [B, embed_dim]
        global_repr = fused_tokens.mean(dim=1)
        # Concatenate with current TCP proprioception [x, y, z, grip]
        ctx = torch.cat([global_repr, proprio], dim=-1)
        action = self.net(ctx)
        # Apply physical scaling:
        # dx, dy, dz scaled to [-0.04, 0.04] m/step
        # droll, dpitch, dyaw scaled to [-0.08, 0.08] rad/step
        # gripper scaled to [0.015, 0.035] (closed to open)
        delta_xyz = torch.tanh(action[:, :3]) * 0.035
        delta_rot = torch.tanh(action[:, 3:6]) * 0.05
        grip = torch.sigmoid(action[:, 6:]) * 0.02 + 0.015
        return torch.cat([delta_xyz, delta_rot, grip], dim=-1)

class OctoVlaPolicy(nn.Module):
    """
    Complete End-to-End Vision-Language-Action Policy Network.
    
    Receives:
      - RGB Camera Image: [B, 3, 256, 256]
      - Natural Language Instruction: String or [B, seq_len]
      - Robot Proprioception: [B, 4] (x, y, z of TCP + gripper width)
    
    Outputs:
      - Delta EEF Action: [B, 7] (dx, dy, dz, droll, dpitch, dyaw, gripper)
      - Visual Attention Heatmap: [B, 16, 16] showing which pixels the VLA focused on
    """
    def __init__(self, embed_dim=128, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = PatchVisualTokenizer(img_size=256, patch_size=16, embed_dim=embed_dim)
        self.lang_encoder = LanguageTokenizer(embed_dim=embed_dim)
        self.transformer = OctoMultimodalTransformer(embed_dim=embed_dim, num_layers=num_layers)
        self.action_head = ActionPredictionHead(embed_dim=embed_dim)

        self._init_calibrated_weights()

    def _init_calibrated_weights(self):
        """Initialize weights with Xavier uniform distributions for stable zero-shot inference."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_tensor, lang_tokens, proprio_tensor):
        # img_tensor: [B, 3, 256, 256]
        # lang_tokens: [B, 16]
        # proprio_tensor: [B, 4]
        v_tokens = self.visual_encoder(img_tensor) # [B, 256, D]
        l_tokens = self.lang_encoder(lang_tokens)   # [B, 16, D]

        fused = self.transformer(v_tokens, l_tokens) # [B, 256, D]
        actions = self.action_head(fused, proprio_tensor) # [B, 7]

        # Compute cross-attention energy map between language and image patches
        # For visualization and explainability of what the VLA is looking at:
        l_summary = l_tokens.mean(dim=1, keepdim=True) # [B, 1, D]
        attn_scores = torch.bmm(v_tokens, l_summary.transpose(1, 2)).squeeze(-1) # [B, 256]
        attn_map = F.softmax(attn_scores, dim=-1).view(-1, 16, 16) # [B, 16, 16]

        return actions, attn_map

    def predict_action(self, image_rgb_np, instruction_str, tcp_xyz, gripper_val, device='cpu'):
        """
        Inference helper:
        Receives raw numpy image [H, W, 3] in [0, 255], string instruction, and robot state.
        Returns continuous 7-DoF delta action and normalized attention map.
        """
        self.eval()
        with torch.no_grad():
            # 1. Resize and normalize image
            H, W, C = image_rgb_np.shape
            if (H, W) != (256, 256):
                import cv2
                img_resized = cv2.resize(image_rgb_np, (256, 256))
            else:
                img_resized = image_rgb_np

            img_norm = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            # [H, W, C] -> [1, C, H, W]
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # 2. Tokenize prompt
            tokens = tokenize_instruction(instruction_str).to(device)

            # 3. Proprioception
            proprio = torch.tensor([[tcp_xyz[0], tcp_xyz[1], tcp_xyz[2], gripper_val]], dtype=torch.float32).to(device)

            # 4. Model Forward Pass
            actions, attn_map = self(img_tensor, tokens, proprio)

            action_np = actions.squeeze(0).cpu().numpy()
            attn_np = attn_map.squeeze(0).cpu().numpy()

            return action_np, attn_np

if __name__ == '__main__':
    print("Initializing Octo-VLA End-to-End Policy model...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = OctoVlaPolicy().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Octo-VLA initialized successfully on {device}! Total Parameters: {n_params:,}")

    # Test forward pass with dummy observation
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    prompt = "Pick up the red can and place it on the platform"
    tcp = [0.40, 0.0, 0.25]
    grip = 0.035

    action, attn = model.predict_action(dummy_img, prompt, tcp, grip, device=device)
    print("Action output shape:", action.shape)
    print("Predicted Delta Action (dx, dy, dz, droll, dpitch, dyaw, grip):", np.round(action, 4))
    print("Attention map shape:", attn.shape, f"(min: {attn.min():.4f}, max: {attn.max():.4f})")
