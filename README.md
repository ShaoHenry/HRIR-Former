# HRIR-Former: Joint HRIR Inpainting

This repository provides the evaluation implementation for the following paper:

**HRIR-Former: Grid-Free Time-Domain Reconstruction of Head-Related Impulse Responses with a Spatially Encoded Transformer**

```bibtex
@misc{xu2026hrirformergridfreetimedomainreconstruction,
      title={HRIR-Former: Grid-Free Time-Domain Reconstruction of Head-Related Impulse Responses with a Spatially Encoded Transformer},
      author={Shaoheng Xu and Chunyi Sun and Jihui Zhang and Amy Bastine and Prasanga N. Samarasinghe and Thushara D. Abhayapala and Hongdong Li},
      year={2026},
      eprint={2603.27998},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.27998},
}
```

This repository currently contains the code required to evaluate the joint HRIR inpainting model from sparse spatial measurements.

**The training code will be released soon.**

## HOW TO INSTALL

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA execution, install the PyTorch build appropriate for your CUDA version following the official PyTorch installation instructions.

## HOW TO DOWNLOAD DATASET AND CHECKPOINT

The dataset and pretrained checkpoints used for evaluation can be downloaded from:

[Google Drive: Dataset and Checkpoints](https://drive.google.com/drive/folders/1wo_otJxaqeGL6N-ADjMVBDuX3L4ygms1?usp=sharing)

After downloading, place the checkpoint files in:

```text
checkpoints_hrtf/
```

The supported numbers of known directions are:

```text
3, 5, 19, 100
```

The corresponding checkpoint files are:

```text
checkpoints_hrtf/checkpoint_3.pth
checkpoints_hrtf/checkpoint_5.pth
checkpoints_hrtf/checkpoint_19.pth
checkpoints_hrtf/checkpoint_100.pth
```

The test script automatically selects the corresponding checkpoint according to `--num_known`.

The MATLAB dataset used for evaluation contains:

```text
HRIR_L_meas
HRIR_R_meas
POS_meas
```

with shapes:

```text
HRIR_L_meas: (n_directions, n_samples, n_subjects)
HRIR_R_meas: (n_directions, n_samples, n_subjects)
POS_meas:    (n_directions, 3, n_subjects)
```

Provide the downloaded dataset path using `--mat_path`.

## HOW TO TEST

Run the evaluation script by specifying the dataset path and the number of known directions.

For example, to evaluate the model with 19 known directions:

```bash
python test_inpaint_joint.py \
  --mat_path /path/to/SONICOM_aligned_withoutMinPhase.mat \
  --batch_size 8 \
  --ckpt checkpoints_hrtf \
  --num_known 19
```

To evaluate the model with 3 known directions:

```bash
python test_inpaint_joint.py \
  --mat_path /path/to/SONICOM_aligned_withoutMinPhase.mat \
  --batch_size 8 \
  --ckpt checkpoints_hrtf \
  --num_known 3
```

Supported values for `--num_known` are:

```text
3, 5, 19, 100
```

CUDA is used by default:

```bash
python test_inpaint_joint.py \
  --mat_path /path/to/SONICOM_aligned_withoutMinPhase.mat \
  --num_known 19 \
  --device cuda
```

To run on CPU:

```bash
python test_inpaint_joint.py \
  --mat_path /path/to/SONICOM_aligned_withoutMinPhase.mat \
  --num_known 19 \
  --device cpu
```

If CUDA is requested but unavailable, the script automatically falls back to CPU.

To save reconstructed HRIRs, visualization figures, and evaluation metrics:

```bash
python test_inpaint_joint.py \
  --mat_path /path/to/SONICOM_aligned_withoutMinPhase.mat \
  --num_known 19 \
  --output_dir outputs/test_num_known_19
```

The evaluation reports:

```text
NMSE (dB)
Cosine distance
ITD error (microseconds)
ILD error (dB)
```

When `--output_dir` is specified, the output directory contains:

```text
outputs/test_num_known_19/
├── metrics.json
├── predictions/
│   ├── subject_1.npz
│   ├── subject_2.npz
│   └── ...
└── visualizations/
    ├── subject_1.png
    ├── subject_2.png
    └── ...
```

Each `.npz` file contains:

```text
prediction
target
mask
```

The `prediction` and `target` arrays contain the de-normalized binaural HRIRs, while `mask` indicates the known and missing spatial directions.

Each PNG visualization contains heatmaps for:

- left-ear ground-truth HRIRs;
- left-ear reconstructed HRIRs;
- left-ear absolute reconstruction error;
- right-ear ground-truth HRIRs;
- right-ear reconstructed HRIRs;
- right-ear absolute reconstruction error.

The visualization title also reports the number of known and missing spatial directions for the evaluated subject.