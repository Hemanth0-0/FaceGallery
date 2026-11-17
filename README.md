# FaceGallery — Reference-Based Group Photo Retrieval

FaceGallery is a computer-vision prototype that uses a single reference face image to retrieve photographs containing the same identity. It combines face detection, deep face embeddings, and cosine-similarity matching to search an image gallery.

## What the project does

1. Loads identity-labelled face images from the LFW `lfw-deepfunneled` dataset.
2. Keeps identities with at least 20 images.
3. Uses InsightFace's `buffalo_l` model for face detection and 512-dimensional face embeddings.
4. Normalizes embeddings so dot products can be used as cosine similarity.
5. Compares a reference-image embedding against indexed gallery embeddings.
6. Retrieves and ranks matching images using a configurable similarity threshold.
7. Creates synthetic multi-face composites from LFW portraits to test multi-face detection and group-photo retrieval logic.

## Pipeline

```text
LFW images
   |
   v
Identity filtering (minimum 20 images/person)
   |
   v
InsightFace buffalo_l
   |-- face detection
   |-- 512-D face embeddings
   |
   v
Normalized embedding gallery + image metadata
   |
   +---------------- Reference image
   |                         |
   |                    Query embedding
   |                         |
   +---------- Cosine similarity
                             |
                     Threshold filtering
                             |
                    Deduplicate by photo
                             |
                    Ranked photo results
```

## Tech stack

- Python
- OpenCV
- InsightFace (`buffalo_l`)
- ONNX Runtime
- NumPy
- scikit-learn (precision/recall calculations)
- Matplotlib (visualization)

**Implementation note:** `faiss-cpu` appeared in the original installation command, but the notebook does not currently use a FAISS index. Similarity search is performed with NumPy matrix-vector multiplication. The notebook's saved configuration uses `CPUExecutionProvider`, despite a comment mentioning switching to CUDA.

## Dataset

The project uses the [Labeled Faces in the Wild (LFW) dataset — Kaggle mirror](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset), specifically the `lfw-deepfunneled` image directory.

The notebook filters the data to identities with at least 20 images. In the saved run, this produced:

| Item | Observed value |
|---|---:|
| Images loaded | 3,023 |
| Identities retained | 62 |
| Face detections / embeddings stored | 3,689 |
| Embedding dimension | 512 |

The number of stored detections is higher than the number of source images. The detection outputs should therefore be inspected for multiple detections per source image before interpreting retrieval metrics.

### Dataset access

The dataset is not included in this repository. Download it from the Kaggle page above and make the folder available to the notebook. The notebook currently expects:

```text
/kaggle/input/datasets/jessicali9530/lfw-dataset/
    lfw-deepfunneled/
        lfw-deepfunneled/
            <person_name>/
                <image>.jpg
```

If you run outside Kaggle, update `LFW_DIR` in the notebook to your local dataset path.

**Dataset rights and attribution:** LFW images were collected from the web and may remain subject to the original image copyright holders. Verify the Kaggle dataset page's current terms and the original LFW usage/citation guidance before redistributing images or using them commercially. This repository therefore does not redistribute the image dataset.

Suggested original dataset citation:

> Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* University of Massachusetts, Amherst, Technical Report 07-49, October 2007.

## Installation

Python 3.9+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

The notebook also installs the dependencies in its first setup cell.

## Run

1. Download the LFW dataset from the Kaggle link.
2. Attach the dataset to a Kaggle notebook, or update `LFW_DIR` to your local path.
3. Open `notebooks/facegallery_reference_retrieval.ipynb`.
4. Run cells from top to bottom.
5. Use `query_person()` for a reference image against the LFW image bank.
6. Run the synthetic composite and event-retrieval cells to exercise the multi-face workflow.

InsightFace downloads the `buffalo_l` model files on first use. Ensure the runtime can access the model download location.

## Experiments recorded in the notebook

### Threshold sweep

The saved notebook reports the following macro-averaged precision and recall over 30 randomly selected query images (random seed 42):

| Similarity threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.981 | 0.997 | 0.989 |
| 0.35 | 0.989 | 0.997 | 0.993 |
| 0.40 | 0.991 | 0.991 | 0.991 |
| 0.45 | 0.993 | 0.988 | 0.990 |
| 0.50 | 0.994 | 0.970 | 0.982 |
| 0.55 | 0.996 | 0.920 | 0.956 |
| 0.60 | 0.996 | 0.805 | 0.891 |

These are the notebook's observed threshold-sweep results, not an independent test-set score. The same 30 queries are reused for all thresholds, so selecting the best threshold from this sweep and reporting its score on those same queries is optimistic. Use a separate validation set to choose a threshold and a held-out test set for final reporting.

### Synthetic group-photo tests

The notebook builds synthetic composites by placing individual LFW portraits side-by-side in a grid. In the saved run:

- 4-person composites: 80 faces placed; 98 detections reported.
- 8-person composites: 160 faces placed; 197 detections reported.
- A synthetic event of 100 composites produced 458 stored face detections.
- Five identities appearing in at least three synthetic event photos were tested for retrieval. The observed precision/recall pairs were 1.00/1.00, 1.00/1.00, 0.97/1.00, 1.00/1.00, and 1.00/1.00.

A detection count greater than the number of faces placed is not a detection rate above 100%; it indicates over-detection, duplicate detections, or false detections. The current composite test counts detections but does not match each detection to a ground-truth face box/identity. These results should be treated as a preliminary pipeline demonstration, not a real-world group-photo benchmark.

## Current limitations and next improvements

- The LFW images are mostly individual face portraits. Synthetic composites do not reproduce real event photographs, occlusion, crowding, motion blur, or varied camera distances.
- The current query functions use the first detected face in a reference image (`faces[0]`).
- Multiple detections per source image need to be validated and deduplicated at the face level.
- The threshold sweep needs separate validation and test identities/images.
- Group detection should be evaluated with face-level ground-truth boxes and identity labels, including precision, recall, and false positives.
- The current code uses NumPy similarity search; FAISS is not yet integrated.
- The current notebook configures ONNX Runtime to use CPU. GPU execution requires configuring and verifying the CUDA provider in the actual runtime.
- Face recognition is sensitive biometric processing. Use it only with appropriate consent, lawful data access, and safeguards; do not use this prototype for consequential decisions.

## Repository structure

```text
FaceGallery/
├── data/
│   └── README.md
├── notebooks/
│   └── facegallery_reference_retrieval.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## License

No license is included for the LFW images or the pretrained InsightFace model. Check the relevant dataset and model terms before redistribution or commercial use. If you want to license your own code, add a separate license only after confirming that it does not imply rights over third-party data or model weights.
