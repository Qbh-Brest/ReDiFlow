# ReDiFlow

Official research implementation of **ReDiFlow: [Full Paper Title]**.

ReDiFlow is a [brief description, e.g., flow-based protein–ligand docking framework]
designed for [main task]. It models [translation, rotation, and torsional updates]
during protein–ligand conformational generation.

> The repository is currently being organized for reproducibility.
> Documentation and pretrained checkpoints will be updated during the review process.

## Overview

The overall workflow consists of:

1. Protein and ligand preprocessing
2. Geometric graph construction
3. Equivariant conformational modeling
4. Pose generation and ranking
5. Evaluation using docking metrics

[Optional: insert one framework figure here]

## Environment
The code was tested with:
- Python 3.10.15
- PyTorch 2.5.1+cu124
- PyTorch Geometric 2.6.1
- CUDA 12.4
- RDKit 2026.03.1

Create the environment using:

```bash
conda env create -f environment.yml
conda activate rediflow
