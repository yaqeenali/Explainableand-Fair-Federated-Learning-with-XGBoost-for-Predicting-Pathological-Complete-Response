"""
Radiomic feature extraction from 3D DCE-MRI tumor volumes using PyRadiomics.

Captures tumor phenotype through:
    - Shape features (3D diameter, sphericity, surface area, ...)
    - First-order statistics (mean, kurtosis, skewness, ...)
    - Texture: GLCM, GLRLM, GLSZM, NGTDM, GLDM

Preprocessing (Section 2.2):
    - Z-score normalisation
    - Isotropic resampling to 1 mm³

Usage:
    python feature_engineering/radiomic_extraction.py \
        --input_dir  /data/mama-mia/nifti \
        --mask_dir   /data/mama-mia/masks \
        --output_csv /data/mama-mia/radiomics.csv \
        --num_workers 8

Reference:
    Ali et al., SPIE Medical Imaging 2026, Proc. SPIE Vol. 13926, 139260Q
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.getLogger("radiomics").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
#  PyRadiomics configuration                                                   #
# --------------------------------------------------------------------------- #

PYRADIOMICS_PARAMS = {
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],   # isotropic 1 mm³ (paper)
        "interpolator": "sitkBSpline",
        "normalize": True,
        "normalizeScale": 100,
        "removeOutliers": 3.0,
        "minimumROIDimensions": 2,
        "minimumROISize": 50,
    },
    "featureClass": {
        "shape":         [],    # 3D shape (diameter, sphericity, ...)
        "firstorder":    [],    # intensity statistics
        "glcm":          [],    # grey-level co-occurrence matrix
        "glrlm":         [],    # grey-level run-length matrix
        "glszm":         [],    # grey-level size zone matrix
        "ngtdm":         [],    # neighbouring grey-tone difference matrix
        "gldm":          [],    # grey-level dependence matrix
    },
}


def build_extractor():
    """Initialise PyRadiomics feature extractor with paper settings."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(PYRADIOMICS_PARAMS)
    return extractor


# --------------------------------------------------------------------------- #
#  Preprocessing                                                               #
# --------------------------------------------------------------------------- #

def preprocess_image(sitk_image):
    """
    Z-score normalise a SimpleITK image (within the image domain).
    Isotropic resampling is handled by PyRadiomics via resampledPixelSpacing.
    """
    arr  = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    mu   = arr.mean()
    std  = arr.std() + 1e-8
    norm = (arr - mu) / std * 100          # scale=100 as in extractor params
    out  = sitk.GetImageFromArray(norm)
    out.CopyInformation(sitk_image)
    return out


# --------------------------------------------------------------------------- #
#  Per-patient extraction                                                      #
# --------------------------------------------------------------------------- #

def extract_patient(args):
    """
    Extract radiomic features for one patient.
    Returns (patient_id, feature_dict) or (patient_id, None) on failure.
    """
    patient_id, image_path, mask_path = args
    try:
        extractor  = build_extractor()
        image      = sitk.ReadImage(str(image_path))
        mask       = sitk.ReadImage(str(mask_path))
        image      = preprocess_image(image)

        result     = extractor.execute(image, mask)

        # Keep only numeric features (drop diagnostics)
        features   = {k: float(v) for k, v in result.items()
                      if not k.startswith("diagnostics_") and np.isscalar(v)}
        features["patient_id"] = patient_id
        return patient_id, features

    except Exception as e:
        print(f"[ERROR] {patient_id}: {e}")
        return patient_id, None


# --------------------------------------------------------------------------- #
#  Main pipeline                                                               #
# --------------------------------------------------------------------------- #

def extract_all(input_dir, mask_dir, output_csv, num_workers=4):
    """
    Extract features for all patients and save to CSV.

    Expected directory structure:
        input_dir/{patient_id}/image.nii.gz
        mask_dir/{patient_id}/mask.nii.gz
    """
    input_dir = Path(input_dir)
    mask_dir  = Path(mask_dir)

    # Build list of (patient_id, image_path, mask_path)
    jobs = []
    for patient_dir in sorted(input_dir.iterdir()):
        pid        = patient_dir.name
        image_path = patient_dir / "image.nii.gz"
        mask_path  = mask_dir / pid / "mask.nii.gz"

        if image_path.exists() and mask_path.exists():
            jobs.append((pid, image_path, mask_path))
        else:
            print(f"[SKIP] {pid}: missing image or mask")

    print(f"Extracting features for {len(jobs)} patients "
          f"using {num_workers} workers ...")

    all_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(extract_patient, j): j[0] for j in jobs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            pid, feats = future.result()
            if feats is not None:
                all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\nExtracted {len(df)} patients, {df.shape[1]-1} features.")
    print(f"Saved to: {output_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="PyRadiomics feature extraction")
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--mask_dir",    required=True)
    parser.add_argument("--output_csv",  required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_all(args.input_dir, args.mask_dir, args.output_csv, args.num_workers)
