# ReDiFlow

**Redistribution-Decoupled Riemannian Flow Matching for Molecular Docking**

ReDiFlow is an inference-time temporal redistribution and decoupling framework for Riemannian flow-based molecular docking. It allows ligand translation, global rotation, and internal torsion to follow different effective generative times while remaining coupled through the same molecular state.

ReDiFlow operates on a pre-trained docking backbone and does not modify the training objective or learned model parameters. The current implementation uses non-uniform schedules to reorganize the sampling trajectory at inference time.

<!-- TODO(release): add the paper, model checkpoint, and dataset links. -->

## Highlights

- Inference-time modification with no additional backbone training.
- Separate effective clocks for translation, rotation, and torsion.
- State-coupled updates on the product space $\mathbb{R}^3 \times \mathrm{SO}(3) \times \mathbb{T}^m$.
- Evaluation on PoseBusters, ASTEX Diverse, DockGen, and the PDBBind time-split benchmark.
- Controlled ablations, five-seed analyses, structural-validity checks, and schedule-discretization tests.

## Method overview

Standard generative docking evolves translation, rotation, and torsion using one shared scalar time. ReDiFlow relaxes this constraint in two stages:

1. **Global temporal redistribution** replaces uniform progression with a non-uniform shared schedule.
2. **DOF-wise temporal decoupling** assigns independent effective times to translation, rotation, and torsion.

At every sampling step, the pre-trained backbone is queried using the same current protein-ligand state. Each degree of freedom uses its own effective time, and the component-wise updates are then applied jointly. The dynamics are therefore temporally decoupled but state-coupled.

## Installation

The exact environment file and tested package versions will be added with the finalized release. The main dependencies used by the current implementation include:

- Python
- PyTorch
- PyTorch Lightning
- e3nn
- RDKit
- PoseBusters

Create the environment using the file provided with the repository:

```bash
conda env create -f <ENVIRONMENT_FILE>
conda activate <ENVIRONMENT_NAME>
```

Alternatively, if the final repository provides a requirements file:

```bash
pip install -r <REQUIREMENTS_FILE>
```

<!-- TODO(release): replace the placeholders with the real filenames and environment name. -->

## Data preparation

The experiments use the following datasets:

- **PoseBusters V1** for backbone training and in-domain evaluation.
- **ASTEX Diverse** for external evaluation.
- **DockGen** for evaluation on unseen binding sites.
- **PDBBind time-split** for evaluation under temporal distribution shift.

The current manuscript reports 426 usable PoseBusters protein-ligand complexes after preprocessing, with 383 used for backbone training and 43 held out for in-domain testing.

### Suggested data layout

Adapt this layout to the paths expected by the final configuration files:

```text
data/
├── raw/
│   ├── posebusters/
│   ├── astex_diverse/
│   ├── dockgen/
│   └── pdbbind_time_split/
├── processed/
└── splits/
```

### Preprocessing summary

The preprocessing described in the manuscript currently includes:

1. Parse the protein and ligand structures.
2. Remove hydrogen atoms from both ligand and receptor structures.
3. Construct the ligand-pocket molecular graph.
4. Use atomic number and chirality as node features.
5. Build distance-based ligand and ligand-receptor edges.
6. Use a 5.0 A atom-level interaction cutoff with at most 8 nearest neighbors per atom.
7. Define the receptor pocket using a 15.0 A radius and up to 24 nearest receptor C-alpha neighbors.
8. Save processed graphs and dataset split lists for training and evaluation.

Temporary command template:

```bash
python <PREPROCESS_ENTRYPOINT> \
  --config <PREPROCESS_CONFIG> \
  --raw-data-dir <RAW_DATA_DIR> \
  --output-dir <PROCESSED_DATA_DIR>
```

<!-- TODO(release): replace this template with the actual preprocessing entrypoint and arguments. -->

## Training

ReDiFlow itself is applied at inference time. The reference docking backbone is an E(3)-equivariant graph network with five stacked equivariant graph convolutional layers and separate tangent-velocity heads for translation, rotation, and torsion.

The training setup reported in the current manuscript is:

| Setting | Value |
| --- | --- |
| Objective | Riemannian flow-matching loss |
| Optimizer | Adam |
| Base learning rate | $1 \times 10^{-3}$ |
| Weight decay | $1 \times 10^{-5}$ |
| Batch size | 1 complex per GPU |
| Maximum epochs | 200 |
| Validation interval | 10 epochs |
| Early-stopping patience | 100 epochs |
| Precision | bf16 mixed precision |
| Reference hardware | One NVIDIA RTX 3060 GPU |

Temporary command template:

```bash
python <TRAIN_ENTRYPOINT> \
  --config <TRAIN_CONFIG> \
  --data-dir <PROCESSED_DATA_DIR> \
  --output-dir <CHECKPOINT_DIR>
```

<!-- TODO(release): add the exact configuration, split file, checkpoint-selection rule, and command used for the reported model. -->

## Inference

Initial ligand poses are generated through random rigid-body and torsional perturbations without native pocket information. The reference protocol generates 40 candidates per complex, applies 10 outer sampling steps, and ranks the resulting poses using a confidence model.

Temporary command template:

```bash
python <INFERENCE_ENTRYPOINT> \
  --config <INFERENCE_CONFIG> \
  --checkpoint <BACKBONE_CHECKPOINT> \
  --dataset <DATASET_NAME> \
  --num-steps 10 \
  --num-samples 40 \
  --seed 42 \
  --output-dir <PREDICTION_DIR>
```

### Default ReDiFlow schedules

Each per-DOF schedule is independently normalized to sum to 1.

| Step $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\Delta t_{\mathrm{tr}}$ | 0.1400 | 0.1300 | 0.1200 | 0.1100 | 0.1000 | 0.1000 | 0.0900 | 0.0800 | 0.0700 | 0.0600 |
| $\Delta t_{\mathrm{rot}}$ | 0.1300 | 0.1300 | 0.1200 | 0.1200 | 0.1100 | 0.1000 | 0.0900 | 0.0800 | 0.0700 | 0.0500 |
| $\Delta t_{\mathrm{tor}}$ | 0.0539 | 0.0630 | 0.0878 | 0.1295 | 0.1657 | 0.1657 | 0.1295 | 0.0878 | 0.0630 | 0.0539 |

Translation and rotation receive larger steps early in sampling for global placement, whereas torsion receives larger steps in the middle of the trajectory for conformational refinement.

## Evaluation

The manuscript reports:

- Ligand heavy-atom RMSD without additional rigid-body alignment.
- Ligand centroid distance.
- Top-1, Top-5, Top-10, and best-of-40 success rates.
- PoseBusters structural-validity checks.

Temporary command template:

```bash
python <EVALUATION_ENTRYPOINT> \
  --predictions <PREDICTION_DIR> \
  --references <REFERENCE_STRUCTURE_DIR> \
  --run-posebusters \
  --output <METRICS_FILE>
```

<!-- TODO(release): document symmetry handling, file formats, failed-complex handling, and the exact PoseBusters configuration. -->

## Reproduction checklist

Before the public release, the following items should be included or documented:

- [ ] Exact environment or dependency-lock file.
- [ ] Dataset download instructions and licenses.
- [ ] Training, validation, and test split lists.
- [ ] Preprocessing entrypoint and expected directory layout.
- [ ] Training and inference configuration files.
- [ ] Backbone and confidence-model checkpoints.
- [ ] Commands used to reproduce each reported table or figure.
- [ ] Random seeds and deterministic settings.
- [ ] Evaluation scripts and PoseBusters configuration.

## Third-party components

This project uses or adapts third-party scientific software and model components. Before release, retain the corresponding copyright notices, licenses, and citations.

| Component | Role in this repository | Upstream source/license | Local modifications |
| --- | --- | --- | --- |
| PyTorch | Model training and inference | `<ADD LINK AND LICENSE>` | None or `<DESCRIBE>` |
| PyTorch Lightning | Training workflow | `<ADD LINK AND LICENSE>` | None or `<DESCRIBE>` |
| e3nn | Equivariant geometric operations | `<ADD LINK AND LICENSE>` | None or `<DESCRIBE>` |
| DiffDock-related components | Confidence scoring or adapted utilities, where applicable | `<ADD EXACT COMMIT AND LICENSE>` | `<DESCRIBE>` |
| Other reused modules | `<DESCRIBE ROLE>` | `<ADD REPOSITORY, COMMIT, AND LICENSE>` | `<DESCRIBE>` |

If a file is copied or substantially adapted from another repository, add a source comment in that file in addition to listing it here.

## Citation

Citation information will be added when the manuscript is publicly available.

```bibtex
@article{rediflow,
  title   = {ReDiFlow: Redistribution-Decoupled Riemannian Flow Matching for Molecular Docking},
  author  = {<AUTHOR LIST>},
  journal = {<VENUE OR PREPRINT>},
  year    = {<YEAR>}
}
```

## License

`<ADD PROJECT LICENSE AFTER CHECKING COMPATIBILITY WITH ALL REUSED COMPONENTS>`

