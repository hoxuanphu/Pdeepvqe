import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from deepvqe import CCM, DecoderBlock, EncoderBlock, FE


def flatten_speaker_embedding(spk_emb):
    """Normalize SpeechBrain embedding shapes to (B, D)."""
    if spk_emb.dim() == 3 and spk_emb.shape[1] == 1:
        spk_emb = spk_emb.squeeze(1)
    elif spk_emb.dim() > 2:
        spk_emb = spk_emb.reshape(spk_emb.shape[0], -1)
    if spk_emb.dim() != 2:
        raise ValueError(f"Expected speaker embedding with shape (B, D), got {tuple(spk_emb.shape)}")
    return spk_emb


class PersonalizedBottleneck(nn.Module):
    def __init__(self, input_size, hidden_size, emb_dim=192):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.film_gamma = nn.Linear(emb_dim, input_size)
        self.film_beta = nn.Linear(emb_dim, input_size)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x, spk_emb):
        """
        x: (B,C,T,F)
        spk_emb: (B, emb_dim)
        """
        spk_emb = flatten_speaker_embedding(spk_emb)
        if spk_emb.shape[0] != x.shape[0]:
            raise ValueError(
                f"Speaker embedding batch size {spk_emb.shape[0]} does not match input batch size {x.shape[0]}"
            )
        if spk_emb.shape[-1] != self.film_gamma.in_features:
            raise ValueError(
                f"Expected speaker embedding dim {self.film_gamma.in_features}, got {spk_emb.shape[-1]}"
            )

        y = rearrange(x, 'b c t f -> b t (c f)')
        y, _ = self.gru(y)
        y = self.fc(y)

        spk_emb = F.normalize(spk_emb, p=2, dim=-1, eps=1e-8)
        spk_emb = spk_emb.to(device=y.device, dtype=y.dtype)
        gamma = self.film_gamma(spk_emb).unsqueeze(1)
        beta = self.film_beta(spk_emb).unsqueeze(1)

        y = gamma * y + beta
        return rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        source="speechbrain/spkrec-ecapa-voxceleb",
        deepvqe_sr=16000,
        ecapa_sr=16000,
        device=None,
    ):
        super().__init__()
        self.source = source
        self.deepvqe_sr = deepvqe_sr
        self.ecapa_sr = ecapa_sr
        self.ecapa = self._load_ecapa(source, device)
        for param in self.ecapa.parameters():
            param.requires_grad = False
        self.ecapa.eval()
        self.resample = self._build_resampler(deepvqe_sr, ecapa_sr)
        ecapa_device = self._ecapa_device()
        if ecapa_device is not None:
            self.resample.to(ecapa_device)

    @staticmethod
    def _load_encoder_classifier():
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError as exc:
                raise ImportError(
                    "PersonalizedDeepVQE requires speechbrain. Install it with `pip install speechbrain`."
                ) from exc
        return EncoderClassifier

    @staticmethod
    def _build_resampler(orig_freq, new_freq):
        if orig_freq == new_freq:
            return nn.Identity()
        try:
            import torchaudio.transforms as T
        except ImportError as exc:
            raise ImportError(
                "Resampling enrollment audio requires torchaudio. Install it with `pip install torchaudio`."
            ) from exc
        return T.Resample(orig_freq=orig_freq, new_freq=new_freq)

    def _load_ecapa(self, source, device):
        EncoderClassifier = self._load_encoder_classifier()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return EncoderClassifier.from_hparams(source=source, run_opts={"device": str(device)})

    def _ecapa_device(self):
        try:
            return next(self.ecapa.parameters()).device
        except StopIteration:
            return None

    def forward(self, enrollment_wav):
        """
        enrollment_wav: (T), (B,T), or (B,C,T)
        returns: L2-normalized speaker embedding (B, 192)
        """
        if enrollment_wav is None:
            raise ValueError("enrollment_wav must be provided when spk_emb is not given")
        if enrollment_wav.dim() == 1:
            enrollment_wav = enrollment_wav.unsqueeze(0)
        elif enrollment_wav.dim() == 3:
            enrollment_wav = enrollment_wav.mean(dim=1)
        if enrollment_wav.dim() != 2:
            raise ValueError(f"Expected enrollment waveform with shape (B, T), got {tuple(enrollment_wav.shape)}")

        if not torch.is_floating_point(enrollment_wav):
            enrollment_wav = enrollment_wav.float()

        device = self._ecapa_device()
        if device is not None:
            enrollment_wav = enrollment_wav.to(device)
        enrollment_wav = self.resample(enrollment_wav)
        spk_emb = self.ecapa.encode_batch(enrollment_wav)
        spk_emb = flatten_speaker_embedding(spk_emb)
        return F.normalize(spk_emb, p=2, dim=-1, eps=1e-8)


class PersonalizedDeepVQE(nn.Module):
    def __init__(
        self,
        emb_dim=192,
        deepvqe_sr=16000,
        ecapa_sr=16000,
        speaker_encoder_source="speechbrain/spkrec-ecapa-voxceleb",
        device=None,
        use_speaker_encoder=True,
    ):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.bottle = PersonalizedBottleneck(128*9, 64*9, emb_dim=emb_dim)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()

        self.speaker_encoder = None
        if use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(
                source=speaker_encoder_source,
                deepvqe_sr=deepvqe_sr,
                ecapa_sr=ecapa_sr,
                device=device,
            )

    def forward(self, x, enrollment_wav=None, spk_emb=None):
        """
        x: mixture STFT, (B,F,T,2)
        enrollment_wav: optional enrollment waveform, (B,T) or (B,C,T)
        spk_emb: optional pre-computed speaker embedding, (B,192)
        """
        if spk_emb is None:
            if enrollment_wav is None:
                raise ValueError("Must provide either enrollment_wav or spk_emb")
            if self.speaker_encoder is None:
                raise ValueError("use_speaker_encoder=False, so spk_emb must be provided")
            with torch.no_grad():
                spk_emb = self.speaker_encoder(enrollment_wav)

        en_x0 = self.fe(x)
        en_x1 = self.enblock1(en_x0)
        en_x2 = self.enblock2(en_x1)
        en_x3 = self.enblock3(en_x2)
        en_x4 = self.enblock4(en_x3)
        en_x5 = self.enblock5(en_x4)

        en_xr = self.bottle(en_x5, spk_emb)

        de_x5 = self.deblock5(en_xr, en_x5)[..., :en_x4.shape[-1]]
        de_x4 = self.deblock4(de_x5, en_x4)[..., :en_x3.shape[-1]]
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x2.shape[-1]]
        de_x2 = self.deblock2(de_x3, en_x2)[..., :en_x1.shape[-1]]
        de_x1 = self.deblock1(de_x2, en_x1)[..., :en_x0.shape[-1]]

        return self.ccm(de_x1, x)


def stft_magnitude(x, eps=1e-12):
    """x: (..., 2) complex-as-real STFT."""
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + eps)


def si_sdr(estimate, target, eps=1e-8):
    """Scale-invariant SDR for waveform tensors shaped (B,T) or (T)."""
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = torch.sum(estimate * target, dim=-1, keepdim=True) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    )
    projection = scale * target
    noise = estimate - projection
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_sdr_loss(estimate, target, eps=1e-8):
    return -si_sdr(estimate, target, eps=eps).mean()


def magnitude_l1_loss(estimate_stft, target_stft):
    return F.l1_loss(stft_magnitude(estimate_stft), stft_magnitude(target_stft))


def speaker_consistency_loss(target_emb, estimate_emb):
    target_emb = F.normalize(flatten_speaker_embedding(target_emb), p=2, dim=-1, eps=1e-8)
    estimate_emb = F.normalize(flatten_speaker_embedding(estimate_emb), p=2, dim=-1, eps=1e-8)
    return 1 - F.cosine_similarity(target_emb, estimate_emb, dim=-1).mean()


def negative_energy_loss(estimate_waveform, estimate_stft=None):
    """Energy suppression loss for absent-speaker samples."""
    loss = estimate_waveform.pow(2).mean()
    if estimate_stft is not None:
        loss = loss + stft_magnitude(estimate_stft).mean()
    return loss


def output_energy_ratio_db(estimate_waveform, mixture_waveform, eps=1e-8):
    """Negative-case attenuation metric. Lower/more negative is better."""
    estimate_energy = estimate_waveform.pow(2).mean(dim=-1)
    mixture_energy = mixture_waveform.pow(2).mean(dim=-1)
    return 10 * torch.log10((estimate_energy + eps) / (mixture_energy + eps))
