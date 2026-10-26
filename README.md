# HieroglyphAI

**Computer vision for exploring ancient Egyptian hieroglyphs in museum photographs**

[Try the web app](https://hieroglyphai.com)

HieroglyphAI is a project I developed to make ancient Egyptian inscriptions more accessible to museum visitors. A visitor can photograph or upload an inscription, select a region of interest, and explore individual hieroglyphs recognized by a two-stage computer vision system. The app includes an introductory learning section as well as more detailed information for users interested in Middle Egyptian.

The project grew out of my independent study of Middle Egyptian and my interest in applying image recognition to real-world artifacts. Museum photographs pose challenges that clean reference images often do not: dim lighting, glass reflections, weathering, varying stone textures, and partially damaged signs.

## How it works

1. **Capture or upload:** A custom camera/upload component lets visitors select and crop a photograph.
2. **Detect:** A Detectron2 **Faster R-CNN R50-FPN** model locates likely hieroglyphs and produces bounding boxes.
3. **Classify:** Detected regions are cropped and passed to a **ResNet-18** classifier, which predicts Gardiner sign classes.
4. **Present:** The Streamlit app connects identified signs to descriptive metadata and learning materials, with a Basic/Advanced experience and a Previous Scans view.

The deployed app uses the V3A detector and V3 classifier. Its inference pipeline includes bounding-box size filtering, padded glyph crops, and confidence-based handling of uncertain predictions. In the current app code, classification is shown only when the top probability is at least **0.75** and exceeds the second-highest probability by at least **0.20**; otherwise it is marked unknown.

**Scope:** HieroglyphAI recognizes individual signs and provides information about them. It is not a complete automatic translation system.

## Development and iteration

I developed multiple generations of both the glyph detector and classifier. Early experiments used curated images and existing museum image sources; subsequent dataset work incorporated photographs I took in museum settings to better represent the conditions in which visitors would use the app. This repository retains training code, dataset materials, evaluation artifacts, and final checkpoints across **V1, V2, and V3/V3A**, rather than only the latest application.

I also tested the app with photographs and physical exhibits, including visits to the Museum of Fine Arts, Boston and the Harvard Museum of the Ancient Near East. Those tests informed both model/dataset improvements and interface decisions intended to make the tool easier for visitors to use.

## Repository guide

| Folder | Contents |
| --- | --- |
| `glyph_det/` | Hieroglyph detection datasets, annotations, training experiments, evaluation results, and final Faster R-CNN checkpoints |
| `glyph_classification/` | Individual-glyph classification datasets, ResNet-18 training scripts, experiments, and checkpoints |
| `glyph_pipeline/` | Standalone scripts and test materials for combining detection and classification |
| `app/` | Streamlit web interface, custom camera component, learning cards, metadata, and production inference pipeline |

Selected final model checkpoints are stored using **Git LFS**, so cloning with LFS enabled is required to obtain their full contents.

## Run locally

The public app is available at **https://hieroglyphai.com**. Local setup is more involved because the project depends on PyTorch, torchvision, **Detectron2**, and a Supabase backend. The dependency list in `app/requirements.txt` reflects the app environment, but platform-specific Detectron2 installation and compatible package versions may require additional setup.

1. Install **Git LFS**, clone this repository, and ensure the LFS objects are present:
   ```bash
   git lfs install
   git clone https://github.com/SK172212/HieroglyphAI.git
   cd HieroglyphAI
   git lfs pull
   ```
2. Set up a compatible Python environment with PyTorch, torchvision, Detectron2, and the dependencies in `app/requirements.txt`.
3. Copy the final model files into the paths expected by `app/pipeline.py`:
   ```bash
   mkdir -p app/models
   cp glyph_det/experiments/v3A/model_final_v3A.pth app/models/model_final_v3A.pth
   cp glyph_classification/classifier_v3/output/resnet18_v3_museum_aug_sqrtweighted_best_val_loss.pth app/models/
   ```
4. Set the `SUPABASE_URL` and `SUPABASE_KEY` environment variables for a **Supabase project you control**, with the required database table and storage bucket configured. **Never commit credentials.** The app's Previous Scans feature depends on this service.
5. From the app directory, run:
   ```bash
   cd app
   streamlit run app.py
   ```

These steps describe the repository's expected inputs; they are **not a claim that a fresh-clone installation has been end-to-end verified** on every platform.

## Key files

- `app/app.py` — visitor-facing Streamlit experience
- `app/pipeline.py` — current detector/classifier inference, postprocessing, and output generation
- `app/camera_component/` — custom capture, upload, and crop interface
- `app/cards/` — introductory learning materials
- `app/metadataU.json` — hieroglyph metadata used by the interface
- `glyph_pipeline/detect_and_classify.py` — earlier standalone combined pipeline

## Notes on the development process

I designed the project around a problem I encountered while studying Middle Egyptian: inscriptions in museums can be difficult for visitors to interpret, even when individual signs are visible. My work included collecting and organizing training images, labeling glyph locations, training and comparing models, inspecting predictions, testing in museum conditions, and revising the interface. I used AI coding assistance during software development and iteratively tested and revised the resulting code.

This repository preserves the experimental work alongside the finished application so readers can trace how the system evolved.
