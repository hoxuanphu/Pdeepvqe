from copy import deepcopy


BASE_CONFIG = {
    "experiment": {
        "name": "pdeepvqe_ecapa_film",
        "seed": 1337,
        "output_dir": "runs/pdeepvqe_ecapa_film",
        "resume_from": None,
    },
    "model": {
        "module": "deepvqe_personalized",
        "class_name": "PersonalizedDeepVQE",
        "emb_dim": 192,
        "deepvqe_sr": 16000,
        "ecapa_sr": 16000,
        "speaker_encoder_source": "speechbrain/spkrec-ecapa-voxceleb",
        "use_speaker_encoder": False,
        "use_precomputed_spk_emb": True,
    },
    "stft": {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "hann",
        "return_complex_as_real": True,
    },
    "data": {
        "sample_rate": 16000,
        "clip_seconds": 4.0,
        "enrollment_seconds": [3.0, 5.0],
        "num_workers": 4,
        "pin_memory": True,
        "train_manifest": "data/manifests/train.jsonl",
        "valid_manifest": "data/manifests/valid.jsonl",
        "test_manifest": "data/manifests/test.jsonl",
        "speaker_disjoint_split": True,
        "target_kind": "reverberant_target",
        "rir_sources": [
            "data/rir/RIRS_NOISES",
            "data/rir/MIT_RIR_Survey",
        ],
        "mixture_ratios": {
            "target_interferer_noise": 0.70,
            "target_noise": 0.15,
            "target_interferer": 0.15,
        },
        "interferer_speakers": [1, 2],
        "curriculum": {
            "enabled": True,
            "warmup_epochs": 5,
            "easy": {
                "snr_db": [10.0, 20.0],
                "sir_db": [10.0, 20.0],
            },
            "hard": {
                "snr_db": [-5.0, 5.0],
                "sir_db": [-5.0, 5.0],
            },
        },
    },
    "enrollment": {
        "vad": {
            "enabled": True,
            "min_speech_ratio": 0.6,
        },
        "short_audio_policy": "repeat",
        "cache": {
            "enabled": True,
            "embedding_dir": "data/cache/ecapa_embeddings",
            "metadata_path": "data/cache/ecapa_embeddings/metadata.jsonl",
            "format": "npy",
            "cache_per_enrollment_segment": True,
            "required_metadata": [
                "speaker_id",
                "utterance_id",
                "sample_rate",
                "duration",
                "embedding_path",
            ],
            "require_different_target_utterance": True,
        },
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999],
        "grad_clip_norm": 5.0,
    },
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "max",
        "monitor": "valid/si_sdri",
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6,
    },
    "training": {
        "device": "cuda",
        "precision": "fp32",
        "batch_size": 8,
        "epochs": 80,
        "early_stopping": {
            "enabled": True,
            "monitor": "valid/si_sdri",
            "mode": "max",
            "patience": 10,
        },
        "sanity_overfit": {
            "enabled": True,
            "num_batches": 2,
            "max_steps": 300,
        },
        "checkpoint": {
            "save_best": True,
            "save_last": True,
            "monitor": "valid/si_sdri",
            "mode": "max",
        },
    },
    "loss": {
        "phase": "phase1_reconstruction",
        "reconstruction": {
            "si_sdr_weight": 0.5,
            "magnitude_l1_weight": 0.5,
        },
        "speaker_consistency": {
            "enabled": False,
            "alpha": 0.03,
            "target_embedding": "target_waveform",
            "eval_ecapa_source": "speechbrain/spkrec-ecapa-voxceleb",
            "eval_ecapa_frozen": True,
            "eval_ecapa_no_grad": False,
            "apply_every_n_batches": 4,
        },
        "negative_case": {
            "enabled": False,
            "sample_ratio": 0.0,
            "waveform_energy_weight": 1.0,
            "magnitude_suppression_weight": 0.5,
            "do_not_use_si_sdr": True,
        },
    },
    "metrics": [
        "si_sdr",
        "si_sdri",
        "sdr",
        "sdri",
        "pesq_wb",
        "stoi",
        "speaker_cosine",
        "eer",
        "negative_output_energy_ratio_db",
    ],
}


PHASE_OVERRIDES = {
    "phase1_reconstruction": {
        "loss": {
            "phase": "phase1_reconstruction",
            "speaker_consistency": {
                "enabled": False,
            },
            "negative_case": {
                "enabled": False,
                "sample_ratio": 0.0,
            },
        },
    },
    "phase2_speaker_consistency": {
        "loss": {
            "phase": "phase2_speaker_consistency",
            "speaker_consistency": {
                "enabled": True,
                "alpha": 0.03,
                "apply_every_n_batches": 4,
            },
            "negative_case": {
                "enabled": False,
                "sample_ratio": 0.0,
            },
        },
        "training": {
            "batch_size": 4,
            "epochs": 40,
        },
    },
    "phase3_negative_case": {
        "loss": {
            "phase": "phase3_negative_case",
            "speaker_consistency": {
                "enabled": True,
                "alpha": 0.01,
                "apply_every_n_batches": 8,
            },
            "negative_case": {
                "enabled": True,
                "sample_ratio": 0.15,
                "waveform_energy_weight": 1.0,
                "magnitude_suppression_weight": 0.5,
                "do_not_use_si_sdr": True,
            },
        },
        "training": {
            "batch_size": 4,
            "epochs": 20,
        },
    },
}


def deep_update(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_config(phase="phase1_reconstruction"):
    if phase not in PHASE_OVERRIDES:
        valid = ", ".join(PHASE_OVERRIDES)
        raise ValueError(f"Unknown phase {phase!r}. Valid phases: {valid}")
    return deep_update(BASE_CONFIG, PHASE_OVERRIDES[phase])


CONFIG = get_config()
