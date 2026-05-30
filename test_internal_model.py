import torch
from deepvqe_internal import PersonalizedDeepVQE_Internal

def test_shapes():
    print("Khởi tạo mô hình PersonalizedDeepVQE_Internal...")
    model = PersonalizedDeepVQE_Internal(hidden_size=576)
    
    # Batch size = 2, Time = 100, Freq = 257 (DeepVQE default n_fft=512)
    mixture_stft = torch.randn(2, 257, 100, 2)
    enrollment_stft = torch.randn(2, 257, 50, 2)
    
    print(f"Input Mixture STFT shape: {mixture_stft.shape}")
    print(f"Input Enrollment STFT shape: {enrollment_stft.shape}")
    
    print("\n--- Pass 1 (Có Enrollment) ---")
    out1 = model(mixture_stft, enrollment_stft)
    print(f"Output shape: {out1.shape}")
    
    print("\n--- Rút trích Vector K bằng tay để Cache ---")
    spk_emb = model.extract_speaker_embedding(enrollment_stft)
    print(f"Cached spk_emb shape: {spk_emb.shape}")
    
    print("\n--- Pass 2 (Dùng Cached Embedding) ---")
    out2 = model(mixture_stft, spk_emb=spk_emb)
    print(f"Output shape (với cache): {out2.shape}")

if __name__ == '__main__':
    test_shapes()
