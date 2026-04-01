"""
2.5D CT-MRI Dataset for Brownian Bridge Diffusion Model.
Converts 3D volumes to 2.5D slices (each slice uses 3 consecutive slices).
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchio as tio
from typing import List, Optional, Tuple
import numpy as np

from utils.histogram import HistogramExtractor


class CT2MRI_25D_Dataset(Dataset):
    """
    2.5D dataset for CT-to-MRI translation.
    Each sample returns 3 consecutive slices from both CT and MRI.

    For a volume with D slices, this produces D samples:
    - Slice i uses slices [i-1, i, i+1] (with padding at boundaries)
    - Input shape: [3, H, W] for both CT and MRI
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
            metadata_csv: Path to metadata CSV
            augment: Apply data augmentation
            flip: Apply flip augmentation
            mni: Use MNI-registered data
            severance_only: Use only Severance data (exclude SynthRAD)
            use_histogram: Compute histogram context
            histogram_bins: Number of histogram bins
            plane: Slice plane ('axial', 'sagittal', 'coronal')
            target_size: Target square size for padding (default: 256)
        """
        self.metadata_csv = metadata_csv
        self.mni = mni
        self.use_histogram = use_histogram
        self.plane = plane
        self.target_size = target_size

        # Load metadata
        df = pd.read_csv(metadata_csv)
        if severance_only:
            df = df[df['source_folder'] != 'Task1-2']

        required_cols = ['subject_id', 'ct_path', 'mri_path']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Build subject list
        self.subjects = []
        for _, row in df.iterrows():
            ct_path = row['ct_path']
            mri_path = row['mri_path']
            subject_id = row['subject_id']

            if os.path.exists(ct_path) and os.path.exists(mri_path):
                self.subjects.append({
                    "id": subject_id,
                    "ct": ct_path,
                    "mr": mri_path
                })
            else:
                print(f"Warning: Missing files for {subject_id}")

        assert len(self.subjects) > 0, f"No valid subjects in {metadata_csv}"
        print(f"Loaded {len(self.subjects)} subjects from {metadata_csv}")

        # Histogram extractor
        if self.use_histogram:
            self.hist_extractor = HistogramExtractor(
                num_bins=histogram_bins,
                value_range=(-1.0, 1.0),
                normalize=True
            )

        # Augmentation
        self.augment = augment
        print(f"augment = {self.augment}, flip = {flip}")
        if augment:
            if flip:
                self.tx = tio.Compose([
                    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                    tio.RandomAffine(
                        scales=(0.95, 1.05),
                        degrees=10,
                        translation=5,
                        isotropic=False,
                        center='image',
                    ),
                ])
            else:
                self.tx = tio.Compose([
                    tio.RandomAffine(
                        scales=(0.95, 1.05),
                        degrees=10,
                        translation=5,
                        isotropic=False,
                        center='image',
                    ),
                ])
        else:
            self.tx = None

        # Build slice index mapping
        # Each subject contributes D slices (where D is volume depth)
        self._build_slice_index()

    def _build_slice_index(self):
        """
        Build index mapping: global_idx -> (subject_idx, slice_idx)
        This allows us to index individual slices across all subjects.
        """
        self.slice_index = []

        # Sample first subject to get dimensions
        first_subj = self.subjects[0]
        if self.mni:
            #ct_path = first_subj["ct"][:-3] + '_mni.pt'
            ct_path = first_subj["ct"]
            sample_ct = torch.load(ct_path).to(torch.float32)[10:186, 12:220, 6:182]
        else:
            sample_ct = torch.load(first_subj["ct"]).to(torch.float32)

        # Assume all subjects have same dimensions (or we'll check each time)
        self.depth = sample_ct.shape[0]  # D dimension (axial slices)
        self.height = sample_ct.shape[1]  # H dimension
        self.width = sample_ct.shape[2]  # W dimension

        # For each subject, add all slices
        for subj_idx in range(len(self.subjects)):
            for slice_idx in range(self.depth):
                self.slice_index.append((subj_idx, slice_idx))

        print(f"Total slices: {len(self.slice_index)} from {len(self.subjects)} subjects")
        print(f"Volume dimensions: D={self.depth}, H={self.height}, W={self.width}")

    def __len__(self):
        return len(self.slice_index)

    @staticmethod
    def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
        """(0,1) -> (-1,1)"""
        return x * 2.0 - 1.0

    def _get_3_slices(self, volume: torch.Tensor, slice_idx: int) -> torch.Tensor:
        """
        Extract 3 consecutive slices centered at slice_idx.

        Args:
            volume: [D, H, W] tensor
            slice_idx: Center slice index

        Returns:
            [3, H, W] tensor with [slice_idx-1, slice_idx, slice_idx+1]
        """
        D, H, W = volume.shape

        # Handle boundaries with replication padding
        if slice_idx == 0:
            # First slice: replicate first slice
            slices = [volume[0], volume[0], volume[1]]
        elif slice_idx == D - 1:
            # Last slice: replicate last slice
            slices = [volume[D-2], volume[D-1], volume[D-1]]
        else:
            # Middle slices: normal case
            slices = [volume[slice_idx-1], volume[slice_idx], volume[slice_idx+1]]

        return torch.stack(slices, dim=0)  # [3, H, W]

    def _pad_to_square(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad tensor to target square size using zero padding.

        Args:
            tensor: [C, H, W] tensor

        Returns:
            [C, target_size, target_size] tensor
        """
        C, H, W = tensor.shape
        target = self.target_size

        # Already correct size
        if H == target and W == target:
            return tensor

        # Calculate padding
        pad_h = target - H
        pad_w = target - W

        # Ensure non-negative
        assert pad_h >= 0 and pad_w >= 0, f"Image size ({H}, {W}) larger than target ({target})"

        # Apply symmetric padding (top, bottom, left, right)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # PyTorch padding: (left, right, top, bottom) for 2D
        padded = torch.nn.functional.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        return padded

    def __getitem__(self, idx):
        """
        Get a 2.5D slice sample.

        Returns:
            dict with keys:
                - 'id': subject_id
                - 'slice_idx': slice index within volume
                - 'ct': CT slices [3, H, W] in [-1, 1]
                - 'mri': MRI slices [3, H, W] in [-1, 1]
                - 'hist_ct': Histogram of CT (optional) [num_bins]
                - 'hist_mri': Histogram of MRI (optional) [num_bins]
        """
        subj_idx, slice_idx = self.slice_index[idx]
        subject = self.subjects[subj_idx]

        # Load full 3D volumes
        if self.mni:
            #ct = torch.load(subject["ct"][:-3] + '_mni.pt').to(torch.float32)[10:186, 12:220, 6:182]   (before 20260302)
            #mri = torch.load(subject["mr"][:-3] + '_mni.pt').to(torch.float32)[10:186, 12:220, 6:182]  (before 20260302)
            ct = torch.load(subject["ct"]).to(torch.float32)[10:186, 12:220, 6:182]
            mri = torch.load(subject["mr"]).to(torch.float32)[10:186, 12:220, 6:182]
        else:
            ct = torch.load(subject["ct"]).to(torch.float32)
            mri = torch.load(subject["mr"]).to(torch.float32)

        # (0,1) -> (-1,1)
        ct = self.to_minus1_1(ct)
        mri = self.to_minus1_1(mri)

        # Apply augmentation (3D-wise)
        if self.tx:
            # Add channel dimension for TorchIO
            ct_aug = ct.unsqueeze(0)  # [1, D, H, W]
            mri_aug = mri.unsqueeze(0)  # [1, D, H, W]

            subj = tio.Subject(
                ct=tio.ScalarImage(tensor=ct_aug),
                mri=tio.ScalarImage(tensor=mri_aug)
            )
            subj = self.tx(subj)
            ct = subj.ct.tensor[0]  # [D, H, W]
            mri = subj.mri.tensor[0]  # [D, H, W]

        # Extract 2.5D slices
        ct_slices = self._get_3_slices(ct, slice_idx)  # [3, H, W]
        mri_slices = self._get_3_slices(mri, slice_idx)  # [3, H, W]

        # Pad to square
        ct_slices = self._pad_to_square(ct_slices)  # [3, target_size, target_size]
        mri_slices = self._pad_to_square(mri_slices)  # [3, target_size, target_size]

        result = {
            "id": subject["id"],
            "slice_idx": slice_idx,
            "ct": ct_slices,
            "mri": mri_slices,
        }

        # Compute histogram context (from full volume or current slices)
        if self.use_histogram:
            # Option 1: Histogram from 3 slices (more local)
            result["hist_ct"] = self.hist_extractor(ct_slices.unsqueeze(0))[0]  # [num_bins]
            result["hist_mri"] = self.hist_extractor(mri_slices.unsqueeze(0))[0]  # [num_bins]

            # Option 2: Histogram from full volume (more global - matches paper)
            # result["hist_ct"] = self.hist_extractor(ct.unsqueeze(0).unsqueeze(0))[0]
            # result["hist_mri"] = self.hist_extractor(mri.unsqueeze(0).unsqueeze(0))[0]

        return result


class CT2MRI_25D_Inference_Dataset(Dataset):
    """
    2.5D dataset for inference.
    Groups slices by volume for easier 3D reconstruction.
    """

    def __init__(self,
                 metadata_csv: str,
                 mni: bool = False,
                 use_histogram: bool = True,
                 histogram_bins: int = 256,
                 plane: str = 'axial',
                 target_size: int = 256):
        """
        Args:
            metadata_csv: Path to metadata CSV
            mni: Use MNI-registered data
            use_histogram: Compute histogram context
            histogram_bins: Number of histogram bins
            plane: Slice plane
            target_size: Target square size for padding (default: 256)
        """
        self.metadata_csv = metadata_csv
        self.mni = mni
        self.use_histogram = use_histogram
        self.plane = plane
        self.target_size = target_size

        # Load metadata
        df = pd.read_csv(metadata_csv)
        required_cols = ['subject_id', 'ct_path', 'mri_path']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Build subject list
        self.subjects = []
        for _, row in df.iterrows():
            ct_path = row['ct_path']
            mri_path = row['mri_path']
            subject_id = row['subject_id']

            if os.path.exists(ct_path) and os.path.exists(mri_path):
                self.subjects.append({
                    "id": subject_id,
                    "ct": ct_path,
                    "mr": mri_path
                })

        assert len(self.subjects) > 0, f"No valid subjects in {metadata_csv}"
        print(f"Loaded {len(self.subjects)} subjects for inference")

        # Histogram extractor
        if self.use_histogram:
            self.hist_extractor = HistogramExtractor(
                num_bins=histogram_bins,
                value_range=(-1.0, 1.0),
                normalize=True
            )

    def __len__(self):
        return len(self.subjects)

    @staticmethod
    def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    def __getitem__(self, idx):
        """
        Get full volume data for inference.

        Returns:
            dict with:
                - 'id': subject_id
                - 'ct': Full CT volume [D, H, W]
                - 'mri': Full MRI volume [D, H, W]
                - 'hist_ct': Histogram (optional)
                - 'hist_mri': Histogram (optional)
        """
        subject = self.subjects[idx]

        # Load volumes
        if self.mni:
            #ct = torch.load(subject["ct"][:-3] + '_mni.pt').to(torch.float32)[10:186, 12:220, 6:182]
            #mri = torch.load(subject["mr"][:-3] + '_mni.pt').to(torch.float32)[10:186, 12:220, 6:182]
            ct = torch.load(subject["ct"]).to(torch.float32)[10:186, 12:220, 6:182]
            mri = torch.load(subject["mr"]).to(torch.float32)[10:186, 12:220, 6:182]
        else:
            ct = torch.load(subject["ct"]).to(torch.float32)
            mri = torch.load(subject["mr"]).to(torch.float32)

        # (0,1) -> (-1,1)
        ct = self.to_minus1_1(ct)
        mri = self.to_minus1_1(mri)

        result = {
            "id": subject["id"],
            "ct": ct,  # [D, H, W]
            "mri": mri,  # [D, H, W]
        }

        if self.use_histogram:
            result["hist_ct"] = self.hist_extractor(ct.unsqueeze(0).unsqueeze(0))[0]
            result["hist_mri"] = self.hist_extractor(mri.unsqueeze(0).unsqueeze(0))[0]

        return result
