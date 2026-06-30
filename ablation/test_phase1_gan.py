"""Smoke test for Phase 1 Adversarial Training (GAN Loss) integration.

Validates:
1. Discriminator & MultiScaleDiscriminator forward pass
2. Loss functions (adversarial_d_loss, adversarial_g_loss, feature_matching_loss)
3. GAN training loop integration in train_ablation (make_model, make_optimizer_scheduler)
4. Checkpoint save/load with GAN state
5. Config backward compatibility (use_gan=False still works)
"""

import sys
import tempfile
from pathlib import Path

# Bootstrap imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ablation.discriminator import (
    Discriminator,
    MultiScaleDiscriminator,
    adversarial_d_loss,
    adversarial_g_loss,
    feature_matching_loss,
)
from ablation.ablation_config import get_model_config_id, get_train_config, deep_update
from ablation.train_ablation import (
    make_model,
    make_optimizer_scheduler,
    save_checkpoint,
    load_checkpoint,
    _run_d,
    make_grad_scaler,
)


def test_discriminator_forward():
    """Test single-scale Discriminator forward pass."""
    print("[TEST] Discriminator forward pass...", end=" ")
    D = Discriminator(in_channels=1, ndf=16)
    x = torch.randn(2, 1, 257, 50)  # [B, 1, F, T]

    # Without features
    out = D(x)
    assert out.dim() == 4, f"Expected 4D output, got {out.dim()}D"
    assert out.shape[0] == 2, f"Batch mismatch: {out.shape[0]}"
    assert out.shape[1] == 1, f"Channel mismatch: {out.shape[1]}"

    # With features
    out, feats = D(x, return_features=True)
    assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"

    # 3D input (auto unsqueeze)
    x3d = torch.randn(2, 257, 50)
    out3d = D(x3d)
    assert out3d.dim() == 4

    params = sum(p.numel() for p in D.parameters())
    print(f"OK ({params/1e3:.1f}K params, output shape: {out.shape})")


def test_multiscale_discriminator():
    """Test MultiScaleDiscriminator forward pass."""
    print("[TEST] MultiScaleDiscriminator forward pass...", end=" ")
    MSD = MultiScaleDiscriminator(num_scales=3, ndf=16)
    x = torch.randn(2, 1, 257, 50)

    # Without features
    outputs = MSD(x)
    assert isinstance(outputs, list), "Expected list output"
    assert len(outputs) == 3, f"Expected 3 scales, got {len(outputs)}"

    # With features
    outputs, all_feats = MSD(x, return_features=True)
    assert len(outputs) == 3
    assert len(all_feats) == 3
    for feats in all_feats:
        assert len(feats) == 4, f"Each scale should have 4 feature layers"

    params = sum(p.numel() for p in MSD.parameters())
    print(f"OK (3 scales, {params/1e3:.1f}K params)")


def test_loss_functions():
    """Test LSGAN loss functions."""
    print("[TEST] Loss functions...", end=" ")

    # Create fake predictions
    pred_real = [torch.randn(2, 1, 8, 6)]
    pred_fake = [torch.randn(2, 1, 8, 6)]

    # D loss
    loss_d = adversarial_d_loss(pred_real, pred_fake)
    assert loss_d.dim() == 0, "D loss should be scalar"
    assert torch.isfinite(loss_d), "D loss should be finite"

    # G loss
    loss_g = adversarial_g_loss(pred_fake)
    assert loss_g.dim() == 0, "G loss should be scalar"
    assert torch.isfinite(loss_g), "G loss should be finite"

    # Multi-scale
    pred_real_ms = [torch.randn(2, 1, 8, 6) for _ in range(3)]
    pred_fake_ms = [torch.randn(2, 1, 8, 6) for _ in range(3)]
    loss_d_ms = adversarial_d_loss(pred_real_ms, pred_fake_ms)
    loss_g_ms = adversarial_g_loss(pred_fake_ms)
    assert torch.isfinite(loss_d_ms) and torch.isfinite(loss_g_ms)

    print(f"OK (D_loss={loss_d:.4f}, G_loss={loss_g:.4f})")


def test_feature_matching_loss():
    """Test Feature Matching loss."""
    print("[TEST] Feature matching loss...", end=" ")

    # 1 scale, 4 layers
    real_feats = [[torch.randn(2, 16, 64, 25) for _ in range(4)]]
    fake_feats = [[torch.randn(2, 16, 64, 25) for _ in range(4)]]

    loss_fm = feature_matching_loss(real_feats, fake_feats)
    assert loss_fm.dim() == 0
    assert torch.isfinite(loss_fm)

    # Multi-scale FM
    real_feats_ms = [[torch.randn(2, 16, 64, 25) for _ in range(4)] for _ in range(3)]
    fake_feats_ms = [[torch.randn(2, 16, 64, 25) for _ in range(4)] for _ in range(3)]
    loss_fm_ms = feature_matching_loss(real_feats_ms, fake_feats_ms)
    assert torch.isfinite(loss_fm_ms)

    print(f"OK (FM_loss={loss_fm:.4f})")


def test_run_d_wrapper():
    """Test _run_d wrapper normalizes output to list format."""
    print("[TEST] _run_d wrapper...", end=" ")

    x = torch.randn(2, 1, 257, 50)

    # Single D
    D = Discriminator()
    result = _run_d(D, x)
    assert isinstance(result, list), "Single D output should be wrapped in list"
    assert len(result) == 1

    result, feats = _run_d(D, x, return_features=True)
    assert isinstance(result, list) and isinstance(feats, list)
    assert len(result) == 1 and len(feats) == 1

    # Multi-Scale D
    MSD = MultiScaleDiscriminator(num_scales=2)
    result = _run_d(MSD, x)
    assert isinstance(result, list)
    assert len(result) == 2

    print("OK")


def test_config_gan_params():
    """Test GAN config parameters exist and have correct defaults."""
    print("[TEST] Config GAN parameters...", end=" ")

    cfg = get_train_config("Baseline")

    # Check defaults
    assert cfg["training"]["use_gan"] == False, "use_gan default should be False"
    assert cfg["training"]["num_d_scales"] == 1
    assert cfg["loss"]["lamda_adv"] == 0.05
    assert cfg["loss"]["lambda_fm"] == 0.0

    # Check override
    gan_cfg = deep_update(cfg, {
        "training": {"use_gan": True, "num_d_scales": 3},
        "loss": {"lamda_adv": 0.1, "lambda_fm": 2.0}
    })
    assert gan_cfg["training"]["use_gan"] == True
    assert gan_cfg["training"]["num_d_scales"] == 3
    assert gan_cfg["loss"]["lamda_adv"] == 0.1
    assert gan_cfg["loss"]["lambda_fm"] == 2.0

    # Backward compat: original config unchanged
    assert cfg["training"]["use_gan"] == False

    print("OK")


def test_d1b_gan_preset():
    """Test that the best current generator can be trained with GAN loss."""
    print("[TEST] D1b GRU768 GAN preset...", end=" ")

    cfg = get_train_config("GAN_D1b_gru768")
    assert get_model_config_id("GAN_D1b_gru768") == "D1b_gru768"
    assert cfg["experiment"]["config_id"] == "GAN_D1b_gru768"
    assert cfg["model"]["gru_hidden"] == 768
    assert cfg["training"]["use_gan"] is True
    assert cfg["training"]["num_d_scales"] == 3
    assert cfg["loss"]["lamda_adv"] == 0.05
    assert cfg["loss"]["lambda_fm"] == 2.0

    print("OK")


def test_mamba_gan_preset():
    """Test that the Phase 2 Mamba generator can be trained with GAN loss."""
    print("[TEST] Mamba GAN preset...", end=" ")

    cfg = get_train_config("GAN_Mamba_b2_h384")
    assert get_model_config_id("GAN_Mamba_b2_h384") == "Mamba_b2_h384"
    assert cfg["experiment"]["config_id"] == "GAN_Mamba_b2_h384"
    assert cfg["model"]["sequence_model"] == "mamba"
    assert cfg["model"]["mamba_blocks"] == 2
    assert cfg["model"]["mamba_hidden"] == 384
    assert cfg["training"]["use_gan"] is True
    assert cfg["training"]["num_d_scales"] == 3
    assert cfg["loss"]["lamda_adv"] == 0.05
    assert cfg["loss"]["lambda_fm"] == 2.0

    print("OK")


def test_make_optimizer_with_gan():
    """Test optimizer/scheduler creation with Discriminator."""
    print("[TEST] Optimizer/Scheduler with GAN...", end=" ")

    cfg = get_train_config("Baseline")
    device = torch.device("cpu")
    model = make_model(cfg, device)

    # Without GAN
    opt, sched = make_optimizer_scheduler(model, cfg)
    assert isinstance(opt, torch.optim.Adam)

    # With GAN
    D = Discriminator().to(device)
    (opt_G, sched_G), (opt_D, sched_D) = make_optimizer_scheduler(model, cfg, model_D=D)
    assert isinstance(opt_G, torch.optim.Adam)
    assert isinstance(opt_D, torch.optim.Adam)

    # D should have different LR
    d_lr = opt_D.param_groups[0]["lr"]
    g_lr = opt_G.param_groups[0]["lr"]
    assert d_lr <= g_lr, f"D lr ({d_lr}) should be <= G lr ({g_lr})"

    print(f"OK (G_lr={g_lr}, D_lr={d_lr})")


def test_checkpoint_save_load_gan():
    """Test checkpoint save/load preserves GAN state."""
    print("[TEST] Checkpoint save/load with GAN...", end=" ")

    cfg = get_train_config("Baseline")
    cfg["training"]["use_gan"] = True
    device = torch.device("cpu")

    model = make_model(cfg, device)
    D = Discriminator().to(device)
    (opt_G, sched_G), (opt_D, sched_D) = make_optimizer_scheduler(model, cfg, model_D=D)
    scaler = make_grad_scaler(enabled=False)
    scaler_D = make_grad_scaler(enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_gan.pt"
        save_checkpoint(
            ckpt_path, model, opt_G, sched_G, cfg, epoch=5, best_metric=0.123,
            bad_epochs=2, scaler=scaler,
            model_D=D, opt_D=opt_D, scaler_D=scaler_D, scheduler_D=sched_D
        )

        # Load into fresh models
        model2 = make_model(cfg, device)
        D2 = Discriminator().to(device)
        (opt_G2, sched_G2), (opt_D2, sched_D2) = make_optimizer_scheduler(model2, cfg, model_D=D2)
        scaler2 = make_grad_scaler(enabled=False)
        scaler_D2 = make_grad_scaler(enabled=False)

        epoch, best, bad = load_checkpoint(
            ckpt_path, model2, opt_G2, sched_G2, device,
            scaler=scaler2, model_D=D2, opt_D=opt_D2,
            scaler_D=scaler_D2, scheduler_D=sched_D2
        )

        assert epoch == 5, f"Epoch mismatch: {epoch}"
        assert abs(best - 0.123) < 1e-6, f"Best metric mismatch: {best}"
        assert bad == 2, f"Bad epochs mismatch: {bad}"

        # Verify D weights match
        for (n1, p1), (n2, p2) in zip(D.named_parameters(), D2.named_parameters()):
            assert torch.allclose(p1, p2), f"D param mismatch: {n1}"

    print("OK")


def test_backward_compatibility():
    """Test that use_gan=False works exactly as before."""
    print("[TEST] Backward compatibility (no GAN)...", end=" ")

    cfg = get_train_config("Baseline")
    assert cfg["training"]["use_gan"] == False

    device = torch.device("cpu")
    model = make_model(cfg, device)
    opt, sched = make_optimizer_scheduler(model, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_no_gan.pt"
        save_checkpoint(ckpt_path, model, opt, sched, cfg, epoch=3, best_metric=0.5)

        model2 = make_model(cfg, device)
        opt2, sched2 = make_optimizer_scheduler(model2, cfg)
        epoch, best, bad = load_checkpoint(ckpt_path, model2, opt2, sched2, device)

        assert epoch == 3
        # Verify checkpoint does not contain D state
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "model_D" not in ckpt
        assert "opt_D" not in ckpt

    print("OK")


def test_gan_gradient_flow():
    """Test that GAN training step produces valid gradients."""
    print("[TEST] GAN gradient flow...", end=" ")

    D = Discriminator(ndf=8)
    G_param = torch.nn.Parameter(torch.randn(2, 1, 257, 50))

    # Fake spectrogram
    fake_mag = G_param.abs() + 1e-12
    real_mag = torch.randn(2, 1, 257, 50).abs() + 1e-12

    # D step
    pred_real = _run_d(D, real_mag.detach())
    pred_fake = _run_d(D, fake_mag.detach())
    loss_D = adversarial_d_loss(pred_real, pred_fake)
    loss_D.backward()

    d_grad_norm = sum(p.grad.norm().item() for p in D.parameters() if p.grad is not None)
    assert d_grad_norm > 0, "D should have gradients"
    assert G_param.grad is None, "G should NOT have gradients during D step"

    # G step (freeze D)
    D.zero_grad()
    for p in D.parameters():
        p.requires_grad_(False)

    pred_fake_g = _run_d(D, fake_mag)
    loss_G = adversarial_g_loss(pred_fake_g)
    loss_G.backward()

    for p in D.parameters():
        p.requires_grad_(True)

    assert G_param.grad is not None, "G should have gradients during G step"
    assert G_param.grad.norm().item() > 0, "G gradient should be non-zero"

    print(f"OK (D_grad_norm={d_grad_norm:.4f}, G_grad_norm={G_param.grad.norm():.4f})")


def main():
    print("=" * 60)
    print("Phase 1: Adversarial Training (GAN Loss) - Verification")
    print("=" * 60)

    tests = [
        test_discriminator_forward,
        test_multiscale_discriminator,
        test_loss_functions,
        test_feature_matching_loss,
        test_run_d_wrapper,
        test_config_gan_params,
        test_d1b_gan_preset,
        test_mamba_gan_preset,
        test_make_optimizer_with_gan,
        test_checkpoint_save_load_gan,
        test_backward_compatibility,
        test_gan_gradient_flow,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ Phase 1 GAN integration is VERIFIED and ready for training!")


if __name__ == "__main__":
    main()
