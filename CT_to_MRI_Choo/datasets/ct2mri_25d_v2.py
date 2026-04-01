"""
2.5D CT-MRI Dataset for Brownian Bridge Diffusion Model.
v2: Volume caching — all 3D volumes are loaded into RAM once at __init__ time.

Changes from v1 (ct2mri_25d.py):
  - __init__: Calls _load_all_volumes() to pre-load every CT & MRI volume into
              self.ct_volumes / self.mri_volumes (List[Tensor[D,H,W]], normalized to [-1,1]).
  - _build_slice_index: Uses shape from cached self.ct_volumes[0] instead of a
                        fresh torch.load() call.
  - __getitem__: Replaces torch.load() disk reads with direct RAM indexing into
                 self.ct_volumes[subj_idx] / self.mri_volumes[subj_idx].

I/O improvement:
  Before (v1): every __getitem__ call reads the full 3D .pt file from disk
               → O(volume_size) disk I/O per training sample.
  After  (v2): full volumes are read once at init, subsequent __getitem__ calls
               just index in-RAM tensors → effectively zero disk I/O during training.

Memory note:
  All volumes are stored as float32 tensors in RAM.  For large datasets consider
  monitoring system memory before enabling caching.  With DataLoader num_workers>0
  the dataset is pickled per worker; shared-memory transfer is handled by the OS
  copy-on-write mechanism, so actual duplication is minimal for read-only tensors.
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchio as tio
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from utils.histogram import HistogramExtractor


class CT2MRI_25D_Dataset(Dataset):
    """
    2.5D dataset for CT-to-MRI translation with in-memory volume caching.

    Each sample returns 3 consecutive slices from both CT and MRI.
    For a volume with D slices, this produces D samples:
      - Slice i uses slices [i-1, i, i+1] (boundary-replicated)
      - Output shape: [3, H, W] for both CT and MRI
    """

    def __init__(self,
                 metadata_csv: str,
                 augment: bool = False,
                 flip: bool = False,
                 mni: bool = False,
                 severance_only: bool = False,
                 use_histogram: bool = True,
                 histogram_bins: int = 256,
                 plane: str = 'axial',
                 target_size: int = 256):
        """
        Args:
            metadata_csv   : Path to metadata CSV (columns: subject_id, ct_path, mri_path)
            augment        : Apply random 3-D affine augmentation
            flip           : Also apply random flip (only when augment=True)
            mni            : Crop to MNI bounding box [10:186, 12:220, 6:182]
            severance_only : Exclude SynthRAD (Task1-2) subjects
            use_histogram  : Compute histogram context vector per sample
            histogram_bins : Number of histogram bins
            plane          : Slice plane (currently only 'axial' is used)
            target_size    : Target square spatial size; slices are zero-padded to
                             [target_size, target_size] if smaller
        """
        self.metadata_csv  = metadata_csv
        self.mni           = mni
        self.use_histogram = use_histogram
        self.plane         = plane
        self.target_size   = target_size

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
        # 2. Histogram extractor
        # ------------------------------------------------------------------
        if self.use_histogram:
            self.hist_extractor = HistogramExtractor(
                num_bins=histogram_bins,
                value_range=(-1.0, 1.0),
                normalize=True
            )

        # ------------------------------------------------------------------
        # 3. Augmentation transforms
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
        # 4. v2: Pre-load all volumes into RAM (volume caching)
        #    Normalization (0,1) → (-1,1) is applied here so __getitem__
        #    never touches the disk again.
        # ------------------------------------------------------------------
        self._load_all_volumes()

        # ------------------------------------------------------------------
        # 5. Build slice-level index (uses cached volume shapes)
        # ------------------------------------------------------------------
        self._build_slice_index()

    # ======================================================================
    # v2: Volume Caching
    # ======================================================================
    def _load_all_volumes(self):
        """
        Pre-load every CT and MRI volume into RAM as float32 tensors,
        normalized to the [-1, 1] range.

        Populates:
            self.ct_volumes  : List[Tensor[D, H, W]]  — all CT  volumes
            self.mri_volumes : List[Tensor[D, H, W]]  — all MRI volumes

        MNI cropping (if self.mni) is also applied here so __getitem__
        only does lightweight in-memory slicing.
        """
        print(f"[v2] Pre-loading {len(self.subjects)} volume pairs into RAM ...")
        self.ct_volumes: List[torch.Tensor]  = []
        self.mri_volumes: List[torch.Tensor] = []

        for subj in tqdm(self.subjects, desc="Caching volumes"):
            # --- Load ---
            ct  = torch.load(subj["ct"]).to(torch.float32)
            mri = torch.load(subj["mr"]).to(torch.float32)

            # --- MNI crop (applied once at load time) ---
            if self.mni:
                ct  = ct [10:186, 12:220, 6:182]
                mri = mri[10:186, 12:220, 6:182]

            # --- Normalize (0,1) → (-1,1) at load time ---
            ct  = self.to_minus1_1(ct)
            mri = self.to_minus1_1(mri)

            self.ct_volumes.append(ct)
            self.mri_volumes.append(mri)

        print(f"[v2] All volumes cached. "
              f"Volume shape (first subject): "
              f"CT={self.ct_volumes[0].shape}, MRI={self.mri_volumes[0].shape}")

    # ======================================================================
    # Slice index
    # ======================================================================
    def _build_slice_index(self):
        """
        Build global index: global_idx → (subject_idx, slice_idx).

        v2: Uses self.ct_volumes[0].shape (already in RAM) instead of
            loading from disk.
        """
        # Infer dimensions from first cached volume
        first_vol = self.ct_volumes[0]
        self.depth  = first_vol.shape[0]  # D (axial slices)
        self.height = first_vol.shape[1]  # H
        self.width  = first_vol.shape[2]  # W

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
    def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
        """(0, 1) → (-1, 1)"""
        return x * 2.0 - 1.0

    def _get_3_slices(self, volume: torch.Tensor, slice_idx: int) -> torch.Tensor:
        """
        Extract 3 consecutive slices centered at slice_idx with boundary replication.

        Args:
            volume    : [D, H, W] tensor
            slice_idx : Center slice index

        Returns:
            [3, H, W] tensor — [slice_idx-1, slice_idx, slice_idx+1]
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
        """
        Symmetrically zero-pad [C, H, W] to [C, target_size, target_size].
        """
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
    # __getitem__  (v2: no disk I/O — uses cached volumes)
    # ======================================================================
    def __getitem__(self, idx):
        """
        Get a 2.5D slice sample.

        v2 change: CT and MRI volumes are read from self.ct_volumes /
        self.mri_volumes (pre-loaded in RAM) instead of torch.load() from disk.
        Normalization is already applied in the cached tensors.

        Returns dict:
            'id'       : subject_id  (str)
            'slice_idx': slice index within volume  (int)
            'ct'       : CT  slices [3, target_size, target_size] in [-1, 1]
            'mri'      : MRI slices [3, target_size, target_size] in [-1, 1]
            'hist_ct'  : CT  histogram [num_bins]  (only if use_histogram=True)
            'hist_mri' : MRI histogram [num_bins]  (only if use_histogram=True)
        """
        subj_idx, slice_idx = self.slice_index[idx]
        subject = self.subjects[subj_idx]

        # v2: Direct RAM access — no torch.load(), no MNI crop (done at cache time)
        ct  = self.ct_volumes[subj_idx].clone()   # [D, H, W], already in [-1,1]
        mri = self.mri_volumes[subj_idx].clone()  # [D, H, W], already in [-1,1]

        # ------------------------------------------------------------------
        # Apply 3-D augmentation (random per call — cannot be pre-computed)
        # ------------------------------------------------------------------
        if self.tx:
            ct_aug  = ct.unsqueeze(0)   # [1, D, H, W]
            mri_aug = mri.unsqueeze(0)  # [1, D, H, W]
            subj = tio.Subject(
                ct=tio.ScalarImage(tensor=ct_aug),
                mri=tio.ScalarImage(tensor=mri_aug)
            )
            subj = self.tx(subj)
            ct  = subj.ct.tensor[0]   # [D, H, W]
            mri = subj.mri.tensor[0]  # [D, H, W]

        # ------------------------------------------------------------------
        # Extract 2.5D slices and pad to square
        # ------------------------------------------------------------------
        ct_slices  = self._get_3_slices(ct,  slice_idx)   # [3, H, W]
        mri_slices = self._get_3_slices(mri, slice_idx)   # [3, H, W]

        ct_slices  = self._pad_to_square(ct_slices)    # [3, target_size, target_size]
        mri_slices = self._pad_to_square(mri_slices)   # [3, target_size, target_size]

        result = {
            "id":        subject["id"],
            "slice_idx": slice_idx,
            "ct":        ct_slices,
            "mri":       mri_slices,
        }

        # ------------------------------------------------------------------
        # Histogram context
        # ------------------------------------------------------------------
        if self.use_histogram:
            # Option 1 (current default): histogram from 3 slices (more local)
            result["hist_ct"]  = self.hist_extractor(ct_slices.unsqueeze(0))[0]
            result["hist_mri"] = self.hist_extractor(mri_slices.unsqueeze(0))[0]

            # Option 2: histogram from full volume (more global — matches paper).
            # v2: This is now computationally free because the full volume is
            #     already in RAM. Uncomment to use instead of Option 1.
            # result["hist_ct"]  = self.hist_extractor(ct.unsqueeze(0).unsqueeze(0))[0]
            # result["hist_mri"] = self.hist_extractor(mri.unsqueeze(0).unsqueeze(0))[0]

        return result


# =============================================================================
# Inference Dataset (v2: volume caching applied here as well)
# =============================================================================
class CT2MRI_25D_Inference_Dataset(Dataset):
    """
    2.5D dataset for inference with in-memory volume caching.
    Groups slices by volume for easy 3-D reconstruction.
    """

    def __init__(self,
                 metadata_csv: str,
                 mni: bool = False,
                 use_histogram: bool = True,
                 histogram_bins: int = 256,
                 plane: str = 'axial',
                 target_size: int = 256):
        self.metadata_csv  = metadata_csv
        self.mni           = mni
        self.use_histogram = use_histogram
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

        if self.use_histogram:
            self.hist_extractor = HistogramExtractor(
                num_bins=histogram_bins,
                value_range=(-1.0, 1.0),
                normalize=True
            )

        # v2: Pre-load all volumes
        self._load_all_volumes()

    def _load_all_volumes(self):
        print(f"[v2 Inference] Pre-loading {len(self.subjects)} volume pairs into RAM ...")
        self.ct_volumes: List[torch.Tensor]  = []
        self.mri_volumes: List[torch.Tensor] = []

        for subj in tqdm(self.subjects, desc="Caching inference volumes"):
            ct  = torch.load(subj["ct"]).to(torch.float32)
            mri = torch.load(subj["mr"]).to(torch.float32)
            if self.mni:
                ct  = ct [10:186, 12:220, 6:182]
                mri = mri[10:186, 12:220, 6:182]
            ct  = ct  * 2.0 - 1.0  # (0,1) → (-1,1)
            mri = mri * 2.0 - 1.0
            self.ct_volumes.append(ct)
            self.mri_volumes.append(mri)

        print(f"[v2 Inference] All volumes cached.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        """
        Returns full volume data for inference.

        v2: Reads from cached RAM tensors instead of disk.
        """
        subject = self.subjects[idx]

        ct  = self.ct_volumes[idx].clone()   # [D, H, W]
        mri = self.mri_volumes[idx].clone()  # [D, H, W]

        result = {
            "id":  subject["id"],
            "ct":  ct,
            "mri": mri,
        }

        if self.use_histogram:
            result["hist_ct"]  = self.hist_extractor(ct.unsqueeze(0).unsqueeze(0))[0]
            result["hist_mri"] = self.hist_extractor(mri.unsqueeze(0).unsqueeze(0))[0]

        return result
