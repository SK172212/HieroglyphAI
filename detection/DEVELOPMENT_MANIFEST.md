# Detection development artifacts

This directory documents three iterations of the hieroglyph object detector, built using Detectron2 Faster R-CNN (ResNet-50 FPN).

## Files to preserve from the development project

- **Training scripts:** `train_glyph_d2_v1.py`, `train_glyph_d2_v2.py`, `train_glyph_d2_v3A.py`.
- **Dataset utilities:** `split_coco.py`, `merge_coco.py`, `load_coco.py`, `check_dataset.py`, `fix_coco_image_sizes.py`, `remove_missing_coco_image.py`, `list_images_in_json.py`.
- **Evaluation and visualization:** `eval_detector.py`, `compare_detector_predictions.py`, `visualize_detector.py`, `visualize_coco.py`, `vis_predict.py`, `predict_one.py`.
- **Annotations:** original V1, V2, V3A COCO JSON annotation files, including the cleaned V3A master file.
- **Splits:** train and validation annotation JSON files for V1, V2 and V3A.
- **Experiment history:** `metrics.json` for each model version.
- **Checkpoints:** final trained `.pth` file for each version, to be added separately using Git LFS or external storage if appropriate.
- **Illustrative results:** selected detector comparison images and a few test examples, subject to image-use permissions.

## Intentionally excluded from a curated portfolio repository

- **Intermediate checkpoints** (`model_0000*.pth`): large, repetitive training snapshots; the final checkpoint and metrics document the model progression more efficiently.
- **Local Python environments and installed Detectron2 source:** third-party dependencies rather than original project work.
- **Bulk generated prediction images and temporary outputs:** reproducible artifacts that would obscure the important examples.
- **Duplicate copies of images across dataset splits:** avoid redundant storage; preserve split definitions in JSON and provide a reproducible dataset layout.
- **Credentials, environment files, caches and OS metadata:** unnecessary and potentially sensitive.

## Evaluation note

Training and validation metrics should be interpreted in the context of the dataset and split used for each experiment. Performance on small held-out sets does not establish generalization to unfamiliar museum inscriptions.

## Upload status

This document and the per-version model descriptions are present in GitHub. The original training scripts, annotations, metrics, images and binary checkpoints still need to be transferred from the supplied archives or the local project. Do not treat this manifest as evidence those files have already been uploaded.
