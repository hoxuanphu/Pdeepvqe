import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from deepvqe import CCM, DecoderBlock, EncoderBlock, FE

class PersonalizedBottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.emb_dim = hidden_size
        
        # CHÚ Ý: Paper chiếu embedding để match với size của flattened features, sau đó concat.
        # Để sát chuẩn paper nhất (với repo này input_size = 128*9 = 1152):
        self.spk_proj1 = nn.Linear(self.emb_dim, input_size)
        self.spk_ln1 = nn.LayerNorm(input_size)
        
        # Mạng giảm chiều sau khi Concat
        self.fusion_proj = nn.Linear(input_size * 2, input_size)
        self.fusion_ln = nn.LayerNorm(input_size)

        # Figure 1 in the paper applies LN after flattening encoder features.
        self.feature_ln = nn.LayerNorm(input_size)

        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Lớp LayerNorm sau GRU (Nơi lấy Internal Embedding)
        self.gru_ln = nn.LayerNorm(hidden_size)
        
        self.fc = nn.Linear(hidden_size, input_size)

    def extract_internal_embedding(self, enrollment_features):
        """Pass 1: Rút trích Vector K từ âm thanh mẫu"""
        self._check_feature_shape(enrollment_features, "enrollment_features")
        B = enrollment_features.size(0)
        # CHUẨN THEO PAPER: Khởi tạo speaker embedding bằng vector 0 và cho đi qua chung 1 đường chiếu (proj)
        zero_spk = torch.zeros(B, self.emb_dim, device=enrollment_features.device, dtype=enrollment_features.dtype)
        
        k = self.spk_ln1(F.elu(self.spk_proj1(zero_spk)))
        k = k.unsqueeze(1).expand(-1, enrollment_features.shape[1], -1)

        enrollment_features = self.feature_ln(enrollment_features)
        x = torch.cat([enrollment_features, k], dim=-1)
        x = self.fusion_ln(F.elu(self.fusion_proj(x)))
        
        y, _ = self.gru1(x)
        y, _ = self.gru2(y)
        y = self.gru_ln(y)
        
        # Average pooling theo trục thời gian (T)
        spk_emb = torch.mean(y, dim=1) # (B, hidden_size)
        return spk_emb

    def forward(self, mixture_features, spk_emb=None):
        """Pass 2: Xử lý âm thanh nhiễu dựa vào Vector K"""
        self._check_feature_shape(mixture_features, "mixture_features")
        if spk_emb is None:
            B = mixture_features.size(0)
            spk_emb = torch.zeros(B, self.emb_dim, device=mixture_features.device, dtype=mixture_features.dtype)
        else:
            spk_emb = self._normalize_spk_emb(spk_emb, mixture_features)

        k = self.spk_ln1(F.elu(self.spk_proj1(spk_emb)))
        k = k.unsqueeze(1).expand(-1, mixture_features.shape[1], -1) 

        mixture_features = self.feature_ln(mixture_features)
        x = torch.cat([mixture_features, k], dim=-1)
        x = self.fusion_ln(F.elu(self.fusion_proj(x)))

        y, _ = self.gru1(x)
        y, _ = self.gru2(y)
        y = self.gru_ln(y)
        
        out = self.fc(y)
        return out

    def _check_feature_shape(self, features, name):
        if features.dim() != 3:
            raise ValueError(f"{name} must have shape (B, T, {self.input_size}), got {tuple(features.shape)}")
        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"{name} last dim must be {self.input_size}, got {features.shape[-1]}. "
                "Check STFT n_fft/frequency bins; this model expects the DeepVQE default path."
            )

    def _normalize_spk_emb(self, spk_emb, reference):
        if spk_emb.dim() != 2:
            raise ValueError(f"spk_emb must have shape (B, {self.emb_dim}), got {tuple(spk_emb.shape)}")
        expected = (reference.shape[0], self.emb_dim)
        if tuple(spk_emb.shape) != expected:
            raise ValueError(f"spk_emb must have shape {expected}, got {tuple(spk_emb.shape)}")
        return spk_emb.to(device=reference.device, dtype=reference.dtype)


class PersonalizedDeepVQE_Internal(nn.Module):
    def __init__(self, hidden_size=576):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)
        
        # Mặc định repo gốc là input_size = 128*9 = 1152
        self.bottle = PersonalizedBottleneck(input_size=128*9, hidden_size=hidden_size)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()

    def encode(self, x):
        """Hàm helper để chạy qua các block Encoder và thu skip connections"""
        en_x0 = self.fe(x)
        en_x1 = self.enblock1(en_x0)
        en_x2 = self.enblock2(en_x1)
        en_x3 = self.enblock3(en_x2)
        en_x4 = self.enblock4(en_x3)
        en_x5 = self.enblock5(en_x4)
        return en_x5, (en_x0, en_x1, en_x2, en_x3, en_x4, en_x5)

    def forward(self, mixture_stft, enrollment_stft=None, spk_emb=None):
        # 1. Đi qua Encoder cho Mixture
        mix_feat, mix_skips = self.encode(mixture_stft)
        mix_feat_flat = rearrange(mix_feat, 'b c t f -> b t (c f)')
        
        # 2. Rút trích Vector K (hoặc dùng cache lúc Inference)
        if spk_emb is None:
            if enrollment_stft is None:
                raise ValueError("Cần truyền enrollment_stft nếu chưa có spk_emb cache")
            
            # GHI CHÚ: Enrollment chỉ cần output cuối (en_x5) để rút trích embedding.
            # Bỏ qua skip connections của enrollment.
            enroll_feat, _ = self.encode(enrollment_stft)
            enroll_feat_flat = rearrange(enroll_feat, 'b c t f -> b t (c f)')
            
            spk_emb = self.bottle.extract_internal_embedding(enroll_feat_flat)
        
        # 3. Lọc tiếng 
        enh_feat = self.bottle(mix_feat_flat, spk_emb)
        
        # 4. Reshape lại cho Decoder sử dụng kích thước gốc của mix_feat
        B, C, T, Freq = mix_feat.shape
        enh_feat = rearrange(enh_feat, 'b t (c f) -> b c t f', c=C, f=Freq)
        
        # 5. Đi qua Decoder kèm skip connections của Mixture
        en_x0, en_x1, en_x2, en_x3, en_x4, en_x5 = mix_skips
        
        de_x5 = self.deblock5(enh_feat, en_x5)[..., :en_x4.shape[-1]]
        de_x4 = self.deblock4(de_x5, en_x4)[..., :en_x3.shape[-1]]
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x2.shape[-1]]
        de_x2 = self.deblock2(de_x3, en_x2)[..., :en_x1.shape[-1]]
        de_x1 = self.deblock1(de_x2, en_x1)[..., :en_x0.shape[-1]]

        return self.ccm(de_x1, mixture_stft)

    def extract_speaker_embedding(self, enrollment_stft):
        """Extract and return the internal speaker embedding for caching at inference."""
        enroll_feat, _ = self.encode(enrollment_stft)
        enroll_feat_flat = rearrange(enroll_feat, 'b c t f -> b t (c f)')
        return self.bottle.extract_internal_embedding(enroll_feat_flat)
