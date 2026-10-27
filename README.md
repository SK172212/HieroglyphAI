# HieroglyphAI

**Computer vision for exploring ancient Egyptian hieroglyphs in museum photographs**

[Try the web app](https://hieroglyphai.com)

HieroglyphAI is a web application I developed to help museum visitors explore ancient Egyptian inscriptions. Visitors can photograph or upload an artifact, crop the area they want to examine, and identify individual hieroglyphs using a two-stage computer vision system. The app provides information about recognized signs, with introductory and more detailed learning materials.

## Why I built HieroglyphAI

My interest in hieroglyphs began during an eighth-grade visit to the Smithsonian National Museum of Natural History in DC. I was fascinated by the ancient Egyptian galleries and remember wanting to know what the inscriptions on the artifacts were saying. The museum placards provided dates and historical background, but I couldn't find explanations of the hieroglyphs themselves.

That visit inspired me to begin teaching myself Middle Egyptian. Later, I began learning Python and machine learning so I could build a tool that would let other museum visitors interact more directly with the artifacts in front of them. HieroglyphAI grew from that idea: using a visitor's own photograph to identify individual signs and learn what they represent.

## How it works

1. **Capture or upload:** A custom camera/upload component lets visitors take or upload a photograph and crop the region of interest.
2. **Detect:** A Detectron2 **Faster R-CNN R50-FPN** model identifies likely hieroglyphs and produces bounding boxes.
3. **Classify:** The detected regions are cropped and passed to a **ResNet-18** classifier, which predicts Gardiner sign classes.
4. **Present:** The Streamlit app connects identified signs to descriptive metadata, with Basic and Advanced modes, learning materials, and a Previous Scans view.

The current application uses the **v3 detector** and **v3 classifier**. Its inference pipeline filters bounding boxes by size, pads glyph crops, and handles uncertain predictions. A classification is displayed only when the highest predicted probability is at least **0.75** and exceeds the second-highest probability by at least **0.20**; otherwise, the sign is marked as unknown.

**Scope:** HieroglyphAI identifies individual signs and provides information about them. It does not automatically translate complete inscriptions.

## Building the datasets

I assembled the datasets from photographs I took inside museums and images from online museum collections. Because the app is intended for use on real artifacts, the data includes differences in lighting, stone textures, carving styles, preservation, and photographic conditions.

For the **detection dataset**, I manually drew all of the bounding boxes around hieroglyphs in the source images. For the **classification dataset**, I initially cropped and labeled individual glyph images by hand, organizing them by Gardiner sign classification.

When preparing data for **v3**, I used the earlier v2 detector and classifier to propose glyph crops and labels more quickly. I then reviewed every proposed result and manually corrected the many crops and classifications that needed adjustment. This let me build on the earlier models while retaining manual quality control over the dataset.

| | Detection | Classification |
| --- | --- | --- |
| **Task** | Locate individual hieroglyphs in larger photographs | Identify the Gardiner class of a cropped sign |
| **Sources** | My museum photographs and online museum collections | My museum photographs and online museum collections |
| **Initial preparation** | Manually drawn bounding boxes | Manually cropped and labeled glyph images |
| **v3 workflow** | v2-assisted proposals, followed by manual review and correction | v2-assisted crops and predictions, followed by manual review and correction |
| **Model** | Faster R-CNN R50-FPN | ResNet-18 |
| **Current version** | v3 | v3 |

The dataset and experiment directories preserve earlier versions as well as later training materials.

## Model development and museum testing

I trained and evaluated successive versions of the detector and classifier, inspected incorrect predictions, and revised the datasets and inference pipeline. The repository retains training code, evaluation artifacts, and model checkpoints from **v1, v2, and v3**, rather than only the final application.

I also tested HieroglyphAI with museum photographs and exhibits, including visits to the Museum of Fine Arts, Boston and the Harvard Museum of the Ancient Near East. These tests helped me identify challenges such as the impact of museum glass casings and lighting that were less apparent when working with cleaner online reference images. 

Drawing on my knowledge of Middle Egyptian, I also designed separate Basic and Advanced modes to make the app accessible to everyday museum visitors while providing more detailed information for users with a deeper interest in the language.


## Repository guide

| Folder | Contents |
| --- | --- |
| `glyph_det/` | Detection datasets, annotations, training experiments, evaluation results, and Faster R-CNN checkpoints |
| `glyph_classification/` | Individual-glyph datasets, ResNet-18 training scripts, experiments, and checkpoints |
| `glyph_pipeline/` | Earlier standalone scripts and test materials combining detection and classification |
| `app/` | Streamlit interface, custom camera component, learning cards, metadata, and current inference pipeline |

Selected final model checkpoints are stored using **Git LFS**. Git LFS is required to download the full checkpoint files rather than their pointer files.

## Run locally

The working public application is available at **https://hieroglyphai.com**. Running the repository locally requires a compatible Python environment with PyTorch, torchvision, **Detectron2**, Streamlit, and other dependencies, as well as a configured Supabase backend. `app/requirements.txt` records application dependencies, but installing Detectron2 may require platform-specific steps.

1. Install **Git LFS**, clone the repository, and download the model checkpoints:

   ```bash
   git lfs install
   git clone https://github.com/SK172212/HieroglyphAI.git
   cd HieroglyphAI
   git lfs pull
   ```

2. Set up a compatible Python environment with PyTorch, torchvision, Detectron2, and the dependencies in `app/requirements.txt`.

3. Place the final checkpoints at the paths expected by `app/pipeline.py`:

   ```bash
   mkdir -p app/models
   cp glyph_det/experiments/v3A/model_final_v3A.pth app/models/model_final_v3A.pth
   cp glyph_classification/classifier_v3/output/resnet18_v3_museum_aug_sqrtweighted_best_val_loss.pth app/models/
   ```

4. Configure the `SUPABASE_URL` and `SUPABASE_KEY` environment variables for a **Supabase project you control**, including the database table and storage bucket required by the application. **Never commit credentials.** The Previous Scans feature uses this backend.

5. Start the app from its directory:

   ```bash
   cd app
   streamlit run app.py
   ```

**Verification note:** I have launched this repository's Streamlit app successfully on macOS using an existing compatible Python environment, the final checkpoints, and configured Supabase variables. A clean installation from a fresh clone has **not** been verified end to end across platforms.

## Key files

- `app/app.py` — visitor-facing Streamlit application
- `app/pipeline.py` — detector/classifier inference and postprocessing
- `app/camera_component/` — custom capture, upload, and crop interface
- `app/cards/` — introductory learning materials
- `app/metadataU.json` — hieroglyph metadata used by the interface
- `glyph_pipeline/detect_and_classify.py` — earlier standalone combined pipeline

## Coding assistance

I used ChatGPT as a coding assistant during development, including generating code from my specifications, troubleshooting errors, and revising implementations. I directed the project's design, assembled and labeled the datasets, conducted model experiments and evaluation, tested the application in museum settings, and reviewed and modified the code to meet the project's requirements.

The repository preserves the experimental work alongside the finished application so readers can examine how the system evolved.
