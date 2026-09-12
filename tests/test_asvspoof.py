"""
Tests for ASVspoof 2019 LA dataset integration.

These tests use synthetic (in-memory) protocol data and temporary FLAC files —
they do NOT require the real ASVspoof dataset to be extracted.
The tests that need real audio are marked with pytest.mark.integration
and are skipped by default.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import VaaniConfig
from src.dataset import (
    ASVspoofDataset,
    ASVspoofRecord,
    parse_asvspoof_protocol,
    check_speaker_overlap,
    LABEL_REAL,
    LABEL_SYNTHETIC,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0, n_mels=64)


def _write_fake_flac(path: Path, sr: int = 16000, duration: float = 1.5) -> None:
    """Write a short sine-wave FLAC file."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr, format="FLAC")


@pytest.fixture
def fake_protocol_file(tmp_path: Path) -> Path:
    """
    Write a minimal protocol file with 6 entries:
    3 bonafide + 3 spoof, spread across 3 speakers.
    """
    content = textwrap.dedent("""\
        LA_0001 LA_T_0000001 - - bonafide
        LA_0001 LA_T_0000002 - - bonafide
        LA_0002 LA_T_0000003 - A01 spoof
        LA_0002 LA_T_0000004 - A02 spoof
        LA_0003 LA_T_0000005 - - bonafide
        LA_0003 LA_T_0000006 - A03 spoof
    """)
    pf = tmp_path / "fake_protocol.txt"
    pf.write_text(content, encoding="utf-8")
    return pf


@pytest.fixture
def fake_audio_dir(tmp_path: Path) -> Path:
    """Create fake FLAC files matching the fake protocol."""
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    for utt_id in [
        "LA_T_0000001", "LA_T_0000002", "LA_T_0000003",
        "LA_T_0000004", "LA_T_0000005", "LA_T_0000006",
    ]:
        _write_fake_flac(audio_dir / f"{utt_id}.flac")
    return audio_dir


@pytest.fixture
def parsed_records(fake_protocol_file, fake_audio_dir) -> List[ASVspoofRecord]:
    return parse_asvspoof_protocol(fake_protocol_file, fake_audio_dir)


# ---------------------------------------------------------------------------
# 1. Protocol Parsing
# ---------------------------------------------------------------------------

class TestProtocolParsing:
    def test_returns_correct_count(self, parsed_records):
        assert len(parsed_records) == 6

    def test_all_fields_present(self, parsed_records):
        for rec in parsed_records:
            assert rec.speaker_id
            assert rec.utterance_id
            assert rec.system_id
            assert rec.label in (LABEL_REAL, LABEL_SYNTHETIC)
            assert isinstance(rec.audio_path, Path)

    def test_utterance_id_matches_filename(self, parsed_records, fake_audio_dir):
        for rec in parsed_records:
            expected = fake_audio_dir / f"{rec.utterance_id}.flac"
            assert rec.audio_path == expected

    def test_speaker_ids_extracted(self, parsed_records):
        speakers = {r.speaker_id for r in parsed_records}
        assert speakers == {"LA_0001", "LA_0002", "LA_0003"}

    def test_attack_ids_extracted(self, parsed_records):
        systems = {r.system_id for r in parsed_records if r.system_id != "-"}
        assert systems == {"A01", "A02", "A03"}

    def test_missing_protocol_raises(self, tmp_path, fake_audio_dir):
        bad_path = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            parse_asvspoof_protocol(bad_path, fake_audio_dir)

    def test_malformed_line_raises(self, tmp_path, fake_audio_dir):
        bad = tmp_path / "bad.txt"
        bad.write_text("LA_0001 LA_T_0001\n", encoding="utf-8")  # only 2 columns
        with pytest.raises(ValueError, match="columns, expected 5"):
            parse_asvspoof_protocol(bad, fake_audio_dir)

    def test_unknown_key_raises(self, tmp_path, fake_audio_dir):
        bad = tmp_path / "bad_key.txt"
        bad.write_text("LA_0001 LA_T_0001 - - genuine\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown key"):
            parse_asvspoof_protocol(bad, fake_audio_dir)

    def test_empty_lines_ignored(self, tmp_path, fake_audio_dir):
        content = "\nLA_0001 LA_T_0000001 - - bonafide\n\n"
        pf = tmp_path / "sparse.txt"
        pf.write_text(content, encoding="utf-8")
        records = parse_asvspoof_protocol(pf, fake_audio_dir)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# 2. Label Mapping: bonafide -> 0, spoof -> 1
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_bonafide_is_zero(self, parsed_records):
        bonafide = [r for r in parsed_records if r.system_id == "-"]
        assert all(r.label == LABEL_REAL for r in bonafide)
        assert LABEL_REAL == 0

    def test_spoof_is_one(self, parsed_records):
        spoof = [r for r in parsed_records if r.system_id != "-"]
        assert all(r.label == LABEL_SYNTHETIC for r in spoof)
        assert LABEL_SYNTHETIC == 1

    def test_label_never_inferred_from_filename(self, tmp_path):
        """
        Protocol says bonafide, even if filename contains 'spoof' or 'A01'.
        Labels come from column 5, not the filename.
        """
        audio_dir = tmp_path / "flac"
        audio_dir.mkdir()
        _write_fake_flac(audio_dir / "LA_T_A01_fake.flac")

        pf = tmp_path / "proto.txt"
        # Utterance named 'LA_T_A01_fake' but protocol says bonafide
        pf.write_text("LA_0001 LA_T_A01_fake - - bonafide\n", encoding="utf-8")
        records = parse_asvspoof_protocol(pf, audio_dir)
        assert records[0].label == LABEL_REAL  # NOT inferred from 'A01' in filename

    def test_count_correct(self, parsed_records):
        bon   = sum(1 for r in parsed_records if r.label == LABEL_REAL)
        spoof = sum(1 for r in parsed_records if r.label == LABEL_SYNTHETIC)
        assert bon   == 3
        assert spoof == 3


# ---------------------------------------------------------------------------
# 3. Audio Path Resolution
# ---------------------------------------------------------------------------

class TestAudioPathResolution:
    def test_paths_point_to_correct_dir(self, parsed_records, fake_audio_dir):
        for rec in parsed_records:
            assert rec.audio_path.parent == fake_audio_dir

    def test_paths_have_flac_extension(self, parsed_records):
        for rec in parsed_records:
            assert rec.audio_path.suffix == ".flac"

    def test_files_exist(self, parsed_records):
        for rec in parsed_records:
            assert rec.audio_path.exists(), f"Missing: {rec.audio_path}"


# ---------------------------------------------------------------------------
# 4. ASVspoofDataset (using mocked protocol and audio)
# ---------------------------------------------------------------------------

class TestASVspoofDataset:
    @pytest.fixture
    def _make_ds(self, tmp_path, config):
        """Build a minimal train dataset with a custom root."""
        # Replicate exact ASVspoof directory layout
        root = tmp_path / "LA"
        proto_dir = root / "ASVspoof2019_LA_cm_protocols"
        audio_dir = root / "ASVspoof2019_LA_train" / "flac"
        proto_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        content = textwrap.dedent("""\
            LA_0001 LA_T_0000001 - - bonafide
            LA_0002 LA_T_0000002 - A01 spoof
            LA_0003 LA_T_0000003 - - bonafide
        """)
        proto_file = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        proto_file.write_text(content, encoding="utf-8")

        for uid in ["LA_T_0000001", "LA_T_0000002", "LA_T_0000003"]:
            _write_fake_flac(audio_dir / f"{uid}.flac")

        return root, config

    def test_len_correct(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        assert len(ds) == 3

    def test_item_shape(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (1, config.n_mels, config.n_frames)
        assert y.ndim == 0  # scalar

    def test_labels_binary(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        for i in range(len(ds)):
            _, y = ds[i]
            assert float(y) in (0.0, 1.0)

    def test_eval_split_forbidden(self, _make_ds):
        root, config = _make_ds
        with pytest.raises(ValueError, match="eval"):
            ASVspoofDataset("eval", config, asvspoof_root=root, verify_files=False)

    def test_invalid_split_forbidden(self, _make_ds):
        root, config = _make_ds
        with pytest.raises(ValueError, match="train.*dev"):
            ASVspoofDataset("test", config, asvspoof_root=root, verify_files=False)

    def test_speaker_ids_property(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        assert ds.speaker_ids == ["LA_0001", "LA_0002", "LA_0003"]

    def test_system_ids_excludes_bonafide_dash(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        assert ds.system_ids == ["A01"]

    def test_num_bonafide_and_spoof(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        assert ds.num_bonafide == 2
        assert ds.num_spoof == 1

    def test_class_weights_positive(self, _make_ds):
        root, config = _make_ds
        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        w = ds.class_weights()
        assert w.shape == (2,)
        assert torch.all(w > 0)


# ---------------------------------------------------------------------------
# 5. Speaker Leakage Detection
# ---------------------------------------------------------------------------

class TestSpeakerOverlap:
    def _make_proto(self, tmp_path, name: str, content: str, split: str, root: Path) -> Path:
        """Helper to create protocol + audio dir."""
        proto_dir = root / "ASVspoof2019_LA_cm_protocols"
        if split == "train":
            audio_dir = root / "ASVspoof2019_LA_train" / "flac"
            proto_name = "ASVspoof2019.LA.cm.train.trn.txt"
        else:
            audio_dir = root / "ASVspoof2019_LA_dev" / "flac"
            proto_name = "ASVspoof2019.LA.cm.dev.trl.txt"
        proto_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        (proto_dir / proto_name).write_text(content, encoding="utf-8")
        # Create a dummy FLAC for each utterance
        for line in content.strip().splitlines():
            uid = line.split()[1]
            _write_fake_flac(audio_dir / f"{uid}.flac")
        return root

    def test_no_overlap_detected(self, tmp_path, config):
        """Disjoint speakers -> no overlap."""
        root = tmp_path / "LA"
        train_content = textwrap.dedent("""\
            LA_0001 LA_T_0000001 - - bonafide
            LA_0002 LA_T_0000002 - A01 spoof
        """)
        dev_content = textwrap.dedent("""\
            LA_0010 LA_D_0000001 - - bonafide
            LA_0011 LA_D_0000002 - A01 spoof
        """)
        self._make_proto(tmp_path, "train", train_content, "train", root)
        self._make_proto(tmp_path, "dev",   dev_content,   "dev",   root)

        ds_train = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        ds_dev   = ASVspoofDataset("dev",   config, asvspoof_root=root, verify_files=False)

        has_overlap, overlap_list = check_speaker_overlap(ds_train, ds_dev)
        assert not has_overlap
        assert overlap_list == []

    def test_overlap_detected(self, tmp_path, config):
        """Shared speaker -> overlap detected."""
        root = tmp_path / "LA_overlap"
        train_content = textwrap.dedent("""\
            LA_0001 LA_T_0000001 - - bonafide
        """)
        dev_content = textwrap.dedent("""\
            LA_0001 LA_D_0000001 - - bonafide
        """)
        self._make_proto(tmp_path, "train", train_content, "train", root)
        self._make_proto(tmp_path, "dev",   dev_content,   "dev",   root)

        ds_train = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        ds_dev   = ASVspoofDataset("dev",   config, asvspoof_root=root, verify_files=False)

        has_overlap, overlap_list = check_speaker_overlap(ds_train, ds_dev)
        assert has_overlap
        assert "LA_0001" in overlap_list


# ---------------------------------------------------------------------------
# 6. Feature Extraction Integration (with real FLAC files)
# ---------------------------------------------------------------------------

class TestFeatureExtractionIntegration:
    def test_full_pipeline_on_fake_flac(self, fake_protocol_file, fake_audio_dir, config):
        """End-to-end: protocol -> audio load -> chunking -> log-Mel -> tensor shape."""
        from src.audio import load_audio, chunk_audio, is_silent
        from src.features import extract_log_mel

        records = parse_asvspoof_protocol(fake_protocol_file, fake_audio_dir)
        # Test first available bonafide + first spoof
        bonafide = next(r for r in records if r.label == LABEL_REAL)
        spoof    = next(r for r in records if r.label == LABEL_SYNTHETIC)

        for rec in [bonafide, spoof]:
            samples, sr = load_audio(rec.audio_path, config)
            chunks = chunk_audio(samples, sr, config, pad_last=True)
            assert len(chunks) > 0

            chunk = next((c for c in chunks if not is_silent(c, config)), chunks[0])
            feat = extract_log_mel(chunk, sr, config)
            assert feat.shape == (config.n_mels, config.n_frames)
            assert feat.dtype == np.float32
            assert not np.any(np.isnan(feat))

    def test_tensor_shape_matches_model_input(self, fake_protocol_file, fake_audio_dir, config):
        """Tensor shape is exactly (1, n_mels, n_frames) as required by Conv2d."""
        from src.audio import load_audio, chunk_audio
        from src.features import extract_log_mel

        records = parse_asvspoof_protocol(fake_protocol_file, fake_audio_dir)
        rec = records[0]
        samples, sr = load_audio(rec.audio_path, config)
        chunks = chunk_audio(samples, sr, config, pad_last=True)
        feat = extract_log_mel(chunks[0], sr, config)
        tensor = torch.from_numpy(feat).unsqueeze(0)
        assert tensor.shape == (1, config.n_mels, config.n_frames)

    def test_train_dev_labels_correct(self, tmp_path, config):
        """ASVspoofDataset returns correct labels from protocol (not filename)."""
        root = tmp_path / "LA"
        proto_dir = root / "ASVspoof2019_LA_cm_protocols"
        audio_dir = root / "ASVspoof2019_LA_train" / "flac"
        proto_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        # Deliberately name files in a 'spooky' way to confirm label is from protocol
        proto_dir.joinpath("ASVspoof2019.LA.cm.train.trn.txt").write_text(
            "LA_0001 LA_T_bonafide_looking_name - - bonafide\n"
            "LA_0002 LA_T_spoof_looking_name - A01 spoof\n",
            encoding="utf-8",
        )
        _write_fake_flac(audio_dir / "LA_T_bonafide_looking_name.flac")
        _write_fake_flac(audio_dir / "LA_T_spoof_looking_name.flac")

        ds = ASVspoofDataset("train", config, asvspoof_root=root, verify_files=False)
        _, y0 = ds[0]
        _, y1 = ds[1]
        assert float(y0) == 0.0  # protocol says bonafide
        assert float(y1) == 1.0  # protocol says spoof
