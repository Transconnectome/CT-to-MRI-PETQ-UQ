"""
2.5D CT-MRI Dataset for Brownian Bridge Diffusion Model.
v3: Official Choo2024 MR histogram approach.

Changes from v2 (ct2mri_25d_v2.py):
  - Histogram source   : MR (target) volume instead of CT (source) slices
  - Histogram scope    : Global whole-volume instead of 3 local slices
  - Histogram bins     : 128  (was 256)
  - Histogram range    : (0.001, 1.0) raw [0,1] space  (was (-1.0, 1.0))
  - Histogram features : 3 — normalized + cumulative + differential  (was 1)
  - Scale              : ×10 applied after normalization  (was none)
  - Histogram shape    : [128, 3, 1]  (was [256])
  - Encoder            : None — raw values passed directly  (was MLP HistogramEncoder)
  - Inference histogram: average over all training subjects  (was per-CT-slice)

Reference:
  CT2MRI/brain_dataset_utils/generate_total_hist_global.py
  CT2MRI/datasets/custom.py lines 96-102  (train=per-subject, test=avg)

Context shape compatibility:
  attention.py already has:
      if context.dim() == 4: context = rearrange(context, 'b c h w -> b (h w) c')
  So [B, 128, 3, 1] → [B, 3, 128] automatically — no attention changes needed.
  context_dim=128 stays valid (dim of each token remains 128).
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torchio as tio
from typing import List, Optional
from tqdm import tqdm


class CT2MRI_25D_Dataset(Dataset):
    """
    2.5D dataset for CT-to-MRI translation with in-memory volume caching.
    v3: Per-subject MR histogram (3-feature, 128-bin, global, Choo2024 official).

    Each sample returns 3 consecutive slices from both CT and MRI plus
    the whole-volume MR histogram [128, 3, 1] for that subject.

    For a volume with D slices, this produces D samples:
      - Slice i uses slices [i-1, i, i+1] (boundary-replicated)
      - Output shape: [3, H, W] for CT and MRI; [128, 3, 1] for hist_mr
    """

    def __init__(self,
                 metadata_csv: str,
                 augment: bool = False,
                 flip: bool = False,
                 mni: bool = False,
                 severance_only: bool = False,
                 plane: str = 'axial',
                 target_size: int = 256):
        """
        Args:
            metadata_csv   : Path to metadata CSV (columns: subject_id, ct_path, mri_path)
            augment        : Apply random 3-D affine augmentation
            flip           : Also apply random flip (only when augment=True)
            mni            : Crop to MNI bounding box [10:186, 12:220, 6:182]
            severance_only : Exclude SynthRAD (Task1-2) subjects
            plane          : Slice plane (currently only 'axial' is used)
            target_size    : Target square spatial size; slices are zero-padded if smaller
        """
        self.metadata_csv = metadata_csv
        self.mni          = mni
        self.plane        = plane
        self.target_size  = target_size

        # ------------------------------------------------------------------
        # 1. Load metadata and build subject list
        # ------------------------------------------------------------------
        df = pd.read_csv(metadata_csv)
        if severance_only:
            df = df[df['source_folder'] != 'Task1-2']

        for col in ['subject_id', 'ct_path', 'mri_path']:
            assert col in df.columns, f"Missing required column: {col}"

        self.subjects: List[dict] = []
        for _, row in df.iterrows():
            ct_path  = row['ct_path']
            mri_path = row['mri_path']
            sid      = row['subject_id']
            if os.path.exists(ct_path) and os.path.exists(mri_path):
                self.subjects.append({"id": sid, "ct": ct_path, "mr": mri_path})
            else:
                print(f"Warning: Missing files for {sid}")

        assert len(self.subjects) > 0, f"No valid subjects in {metadata_csv}"
        print(f"Loaded {len(self.subjects)} subjects from {metadata_csv}")

        # ------------------------------------------------------------------
        # 2. Augmentation transforms
        # ------------------------------------------------------------------
        self.augment = augment
        print(f"augment = {self.augment}, flip = {flip}")
        if augment:
            transforms = []
            if flip:
                transforms.append(tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5))
            transforms.append(
                tio.RandomAffine(
                    scales=(0.95, 1.05),
                    degrees=10,
                    translation=5,
                    isotropic=False,
                    center='image',
                )
            )
            self.tx = tio.Compose(transforms)
        else:
            self.tx = None

        # ------------------------------------------------------------------
        # 3. v2: Pre-load all volumes into RAM (volume caching)
        # ------------------------------------------------------------------
        self._load_all_volumes()

        # ------------------------------------------------------------------
        # 4. v3: Pre-compute MR histograms (global, whole-volume, Choo2024)
        #    Must come after _load_all_volumes() (needs self.mri_volumes).
        # ------------------------------------------------------------------
        self._compute_mr_histograms()

        # ------------------------------------------------------------------
        # 5. Build slice-level index (uses cached volume shapes)
        # ------------------------------------------------------------------
        self._build_slice_index()

    # ======================================================================
    # Volume Caching (same as v2)
    # ======================================================================
    def _load_all_volumes(self):
        """
        Pre-load every CT and MRI volume into RAM as float32 tensors,
        normalized to the [-1, 1] range.

        Populates:
            self.ct_volumes  : List[Tensor[D, H, W]]
            self.mri_volumes : List[Tensor[D, H, W]]
        """
        print(f"[v3] Pre-loading {len(self.subjects)} volume pairs into RAM ...")
        self.ct_volumes: List[torch.Tensor]  = []
        self.mri_volumes: List[torch.Tensor] = []

        for subj in tqdm(self.subjects, desc="Caching volumes"):
            ct  = torch.load(subj["ct"]).to(torch.float32)
            mri = torch.load(subj["mr"]).to(torch.float32)

            if self.mni:
                ct  = ct [10:186, 12:220, 6:182]
                mri = mri[10:186, 12:220, 6:182]

            # Normalize (0,1) → (-1,1) at load time
            ct  = self._to_minus1_1(ct)
            mri = self._to_minus1_1(mri)

            self.ct_volumes.append(ct)
            self.mri_volumes.append(mri)

        print(f"[v3] All volumes cached. "
              f"Volume shape (first subject): "
              f"CT={self.ct_volumes[0].shape}, MRI={self.mri_volumes[0].shape}")

    # ======================================================================
    # v3: MR Histogram Pre-computation  (Choo2024 official formula)
    # ======================================================================
    def _compute_mr_histograms(self):
        """
        Pre-compute official 3-feature MR histograms for each subject.

        Mirrors Choo2024: generate_total_hist_global.py create_hdf5_dataset()

        Formula:
          - mri_vol is stored in [-1, 1]; convert to [0, 1] first (official
            data is pre-normalized to [0, 1]).
          - Feature 1 (norm_hist)  : np.histogram(mri_01, bins=128, range=(0.001,1.0))
                                     normalized by sum, then × scale (10)
          - Feature 2 (cum_hist)   : cumsum of norm_hist
          - Feature 3 (diff_hist)  : np.diff(norm_hist), first element prepended,
                                     then × scale (10)
          - Combined shape          : (128, 3, 1)  — matches official HDF5 storage

        Populates:
            self.mr_histograms  : List[Tensor[128, 3, 1]]  — one per subject
            self.avg_histogram  : Tensor[128, 3, 1]  — mean over all subjects
                                  (for use by CT2MRI_25D_Inference_Dataset)
        """
        print("[v3] Pre-computing official 3-feature MR histograms ...")
        num_bins = 128
        scale    = 10
        self.mr_histograms: List[torch.Tensor] = []

        for mri_vol in tqdm(self.mri_volumes, desc="MR histograms"):
            # mri_vol : [D, H, W], range [-1, 1]
            # Convert [-1, 1] → [0, 1]  (official data lives in [0, 1])
            mri_01 = ((mri_vol + 1.0) / 2.0).numpy().flatten()

            # Feature 1: normalized histogram (×scale)
            # Official: histograms / histograms.sum(keepdims=True) — no epsilon
            hist, _ = np.histogram(mri_01, bins=num_bins, range=(0.001, 1.0))
            norm_hist = hist / hist.sum(keepdims=True) * scale  # shape (128,)

            # Feature 2: cumulative histogram
            cum_hist = np.cumsum(norm_hist)  # shape (128,)

            # Feature 3: differential histogram (×scale)
            diff_hist = np.diff(norm_hist)                    # shape (127,)
            diff_hist = np.insert(diff_hist, 0, diff_hist[0]) * scale  # shape (128,)

            # Stack → (128, 3) → (128, 3, 1)  [matches official HDF5 storage shape]
            combined = np.stack([norm_hist, cum_hist, diff_hist], axis=1)  # (128, 3)
            combined = combined[:, :, np.newaxis]                          # (128, 3, 1)
            self.mr_histograms.append(torch.from_numpy(combined).float())

        # Average histogram: mean over all training subjects
        # Mirrors: create_hdf5_dataset_avg() in generate_total_hist_global.py
        self.avg_histogram = torch.stack(self.mr_histograms).mean(dim=0)  # [128, 3, 1]

        print(f"[v3] MR histograms computed. "
              f"Shape: {self.mr_histograms[0].shape}, "
              f"avg_histogram shape: {self.avg_histogram.shape}")

    # ======================================================================
    # Slice index
    # ======================================================================
    def _build_slice_index(self):
        """
        Build global index: global_idx → (subject_idx, slice_idx).
        """
        first_vol = self.ct_volumes[0]
        self.depth  = first_vol.shape[0]
        self.height = first_vol.shape[1]
        self.width  = first_vol.shape[2]

        self.slice_index = []
        for subj_idx in range(len(self.subjects)):
            for slice_idx in range(self.depth):
                self.slice_index.append((subj_idx, slice_idx))

        print(f"Total slices : {len(self.slice_index)} "
              f"from {len(self.subjects)} subjects")
        print(f"Volume dims  : D={self.depth}, H={self.height}, W={self.width}")

    # ======================================================================
    # Helpers
    # ======================================================================
    def __len__(self):
        return len(self.slice_index)

    @staticmethod
    def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
        """(0, 1) → (-1, 1)"""
        return x * 2.0 - 1.0

    def _get_3_slices(self, volume: torch.Tensor, slice_idx: int) -> torch.Tensor:
        """
        Extract 3 consecutive slices centered at slice_idx with boundary replication.

        Returns: [3, H, W]
        """
        D = volume.shape[0]
        if slice_idx == 0:
            slices = [volume[0], volume[0], volume[1]]
        elif slice_idx == D - 1:
            slices = [volume[D - 2], volume[D - 1], volume[D - 1]]
        else:
            slices = [volume[slice_idx - 1], volume[slice_idx], volume[slice_idx + 1]]
        return torch.stack(slices, dim=0)  # [3, H, W]

    def _pad_to_square(self, tensor: torch.Tensor) -> torch.Tensor:
        """Symmetrically zero-pad [C, H, W] to [C, target_size, target_size]."""
        C, H, W = tensor.shape
        target = self.target_size

        if H == target and W == target:
            return tensor

        pad_h = target - H
        pad_w = target - W
        assert pad_h >= 0 and pad_w >= 0, \
            f"Image ({H},{W}) larger than target ({target})"

        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        return torch.nn.functional.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

    # ======================================================================
    # __getitem__  (v3: MR histogram, no HistogramExtractor)
    # ======================================================================
    def __getitem__(self, idx):
        """
        Get a 2.5D slice sample.

        Returns dict:
            'id'       : subject_id  (str)
            'slice_idx': slice index within volume  (int)
            'ct'       : CT  slices [3, target_size, target_size] in [-1, 1]
            'mri'      : MRI slices [3, target_size, target_size] in [-1, 1]
            'hist_mr'  : whole-volume MR histogram [128, 3, 1]
                         shape [B, 128, 3, 1] in batch; attention.py auto-rearranges
                         to [B, 3, 128] via `if context.dim() == 4` branch.
        """
        subj_idx, slice_idx = self.slice_index[idx]
        subject = self.subjects[subj_idx]

        ct  = self.ct_volumes[subj_idx].clone()   # [D, H, W]
        mri = self.mri_volumes[subj_idx].clone()  # [D, H, W]

        # ------------------------------------------------------------------
        # Apply 3-D augmentation (random per call — cannot be pre-computed)
        # ------------------------------------------------------------------
        if self.tx:
            ct_aug  = ct.unsqueeze(0)
            mri_aug = mri.unsqueeze(0)
            subj = tio.Subject(
                ct=tio.ScalarImage(tensor=ct_aug),
                mri=tio.ScalarImage(tensor=mri_aug)
            )
            subj = self.tx(subj)
            ct  = subj.ct.tensor[0]
            mri = subj.mri.tensor[0]

        # ------------------------------------------------------------------
        # Extract 2.5D slices and pad to square
        # ------------------------------------------------------------------
        ct_slices  = self._get_3_slices(ct,  slice_idx)
        mri_slices = self._get_3_slices(mri, slice_idx)

        ct_slices  = self._pad_to_square(ct_slices)
        mri_slices = self._pad_to_square(mri_slices)

        # v3: per-subject MR histogram (global, whole-volume)
        hist = self.mr_histograms[subj_idx].clone()  # [128, 3, 1]

        return {
            "id":        subject["id"],
            "slice_idx": slice_idx,
            "ct":        ct_slices,
            "mri":       mri_slices,
            "hist_mr":   hist,          # [128, 3, 1]
        }


# =============================================================================
# Inference Dataset (v3: uses avg_histogram from training set)
# =============================================================================
class CT2MRI_25D_Inference_Dataset(Dataset):
    """
    2.5D dataset for inference with in-memory volume caching.
    v3: Uses average MR histogram from training set for all inference subjects.

    This mirrors Choo2024 create_hdf5_dataset_avg() behavior:
    at test time, the per-subject MR histogram is unavailable (no ground-truth MRI),
    so every subject receives the training-set average histogram.

    Reference: CT2MRI/datasets/custom.py lines 96-102
    """

    def __init__(self,
                 metadata_csv: str,
                 avg_histogram: torch.Tensor,
                 mni: bool = False,
                 plane: str = 'axial',
                 target_size: int = 256):
        """
        Args:
            metadata_csv   : Path to metadata CSV (columns: subject_id, ct_path, mri_path)
            avg_histogram  : [128, 3, 1] — average MR histogram from training set.
                             Computed by CT2MRI_25D_Dataset.avg_histogram after init.
                             All inference subjects receive this same histogram.
            mni            : Crop to MNI bounding box [10:186, 12:220, 6:182]
            plane          : Slice plane
            target_size    : Target square spatial size
        """
        assert avg_histogram.shape == (128, 3, 1), \
            f"avg_histogram must be [128, 3, 1], got {avg_histogram.shape}"

        self.metadata_csv  = metadata_csv
        self.avg_histogram = avg_histogram  # [128, 3, 1]
        self.mni           = mni
        self.plane         = plane
        self.target_size   = target_size

        df = pd.read_csv(metadata_csv)
        for col in ['subject_id', 'ct_path', 'mri_path']:
            assert col in df.columns, f"Missing required column: {col}"

        self.subjects: List[dict] = []
        for _, row in df.iterrows():
            ct_path  = row['ct_path']
            mri_path = row['mri_path']
            sid      = row['subject_id']
            if os.path.exists(ct_path) and os.path.exists(mri_path):
                self.subjects.append({"id": sid, "ct": ct_path, "mr": mri_path})

        assert len(self.subjects) > 0, f"No valid subjects in {metadata_csv}"
        print(f"Loaded {len(self.subjects)} subjects for inference")

        self._load_all_volumes()

    def _load_all_volumes(self):
        print(f"[v3 Inference] Pre-loading {len(self.subjects)} volume pairs into RAM ...")
        self.ct_volumes: List[torch.Tensor]  = []
        self.mri_volumes: List[torch.Tensor] = []

        for subj in tqdm(self.subjects, desc="Caching inference volumes"):
            ct  = torch.load(subj["ct"]).to(torch.float32)
            mri = torch.load(subj["mr"]).to(torch.float32)
            if self.mni:
                ct  = ct [10:186, 12:220, 6:182]
                mri = mri[10:186, 12:220, 6:182]
            ct  = ct  * 2.0 - 1.0
            mri = mri * 2.0 - 1.0
            self.ct_volumes.append(ct)
            self.mri_volumes.append(mri)

        print(f"[v3 Inference] All volumes cached.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        """
        Returns full volume data for inference.

        Returns dict:
            'id'      : subject_id  (str)
            'ct'      : CT volume  [D, H, W] in [-1, 1]
            'mri'     : MRI volume [D, H, W] in [-1, 1]  (ground truth, for evaluation)
            'hist_mr' : avg MR histogram [128, 3, 1]  — same for every subject
        """
        subject = self.subjects[idx]

        ct  = self.ct_volumes[idx].clone()
        mri = self.mri_volumes[idx].clone()

        # v3: every inference subject gets the training-set average histogram
        hist = self.avg_histogram.clone()  # [128, 3, 1]

        return {
            "id":      subject["id"],
            "ct":      ct,
            "mri":     mri,
            "hist_mr": hist,  # [128, 3, 1]
        }
