"""Unit tests for the fMRI CSI arithmetic on synthetic in-memory images (no atlas/network).

The ROI-mask fetch and first-level GLM require nilearn + real preprocessed data and are exercised
on the user's compute, not here. Skipped where nibabel/nilearn are absent (e.g. CI).
"""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from spikeprint.fmri import compute_csi, roi_mean, subject_csi  # noqa: E402


def _img(arr):
    return nib.Nifti1Image(np.asarray(arr, dtype="float32"), affine=np.eye(4))


def test_compute_csi_is_vmpfc_minus_dlpfc():
    assert compute_csi(3.0, 1.0) == 2.0


def test_roi_mean_and_subject_csi_on_synthetic_volume():
    stat = np.zeros((4, 4, 4), dtype="float32")
    stat[0, 0, 0] = 10.0  # "vmPFC" voxel
    stat[3, 3, 3] = 2.0   # "dlPFC" voxel
    vm = np.zeros((4, 4, 4), dtype="int8")
    vm[0, 0, 0] = 1
    dl = np.zeros((4, 4, 4), dtype="int8")
    dl[3, 3, 3] = 1
    stat_img, vm_img, dl_img = _img(stat), _img(vm), _img(dl)
    assert roi_mean(stat_img, vm_img) == 10.0
    assert roi_mean(stat_img, dl_img) == 2.0
    assert subject_csi(stat_img, vm_img, dl_img) == 8.0
