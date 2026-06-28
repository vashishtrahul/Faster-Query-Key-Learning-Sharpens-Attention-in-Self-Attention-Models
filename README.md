# Faster Query–Key Learning Sharpens Attention in Self-Attention Models

**Official code repository for our ICML 2026 paper.**

[Paper](https://openreview.net/pdf?id=yESGs2JgSu) · [Project page](https://github.com/vashishtrahul/Faster-Query-Key-Learning-Sharpens-Attention-in-Self-Attention-Models)

## Overview

Self-attention contains circuit parameters with distinct roles: query and key parameters determine **which tokens model attends**, while value and output parameters maps **attended representations to predictions**.

This work studies a optimization intervention theoretically and empirically:

> **What happens when query and key parameters learn faster than the remaining parameters?**

We investigate how the relative learning rates of query-key and output-value parameters affect:

* the sharpness of learned attention;
* optimization and learning dynamics;
* attention alignment with task-relevant tokens;
* downstream task performance.

The repository includes controlled synthetic experiments, theoretical-dynamics simulations, and experiments on HateXplain, subject–verb agreement, and SQuAD.

## Main idea

Let $$\eta_{QK}$$ denote the learning rate for query-key circuit parameters, and let $$\eta_{OV}$$ denote the learning rate for output-value circuit parameters. The standard baseline uses approximately matched learning rates: $$\eta_{QK} = \eta_{OV}$$. Our faster-QK setting instead uses $$\eta_{QK} > \eta_{OV}$$. The experiments examine how varying the ratio $$r = \frac{\eta_{QK}}{\eta_{OV}}$$

changes the attention distributions and behavior learned by self-attention models.

## Repository structure

| Directory             | Description                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `Fig2-Heatmaps/`      | Controlled experiments and attention visualizations associated with Figure 2.                         |
| `FIg-3-Code/`         | Fixed self-attention experiments for the collapsed and factorized settings in Figure 3.               |
| `Fig-4 Code/`         | Simulations and analyses associated with Figure 4.                                                    |
| `HateXplain Dataset/` | Text-classification experiments comparing faster-QK and matched-learning-rate settings on HateXplain. |
| `SVA-Dataset/`        | Subject–verb agreement training, evaluation, and attention-analysis experiments.                      |
| `Squad-Dataset/`      | Question-answering experiments and attention heatmaps on SQuAD.                                       |

The current release is notebook-first: the notebooks preserve the original experimental and analysis workflows used during the project.

## Suggested reading order

For a quick understanding of the project:

1. Start with `Fig2-Heatmaps/` for the basic experimental intuition.
2. Continue with `FIg-3-Code/` for the controlled fixed-attention analysis.
3. Inspect `Fig-4 Code/` for the learning-dynamics simulations.
4. Compare the faster-QK and baseline notebooks in the dataset directories.

For each downstream task, notebooks containing `Faster` or `Faster_QK` implement the faster query/key condition. Notebooks containing `Baseline`, `Same`, or `Same_LR` implement the corresponding matched-learning-rate comparison.

## Installation

Clone the repository:

```bash
git clone https://github.com/vashishtrahul/Faster-Query-Key-Learning-Sharpens-Attention-in-Self-Attention-Models.git
cd Faster-Query-Key-Learning-Sharpens-Attention-in-Self-Attention-Models
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install \
    jupyterlab \
    torch \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    scikit-learn \
    transformers \
    datasets \
    einops \
    fancy-einsum \
    spacy \
    ekphrasis
```

Some notebooks use the historical Easy-Transformer package and may contain additional experiment-specific installation cells.

Launch Jupyter:

```bash
jupyter lab
```

## Data

The experiments use the following datasets and resources:

* **HateXplain** for text classification and rationale-based attention analysis;
* **subject–verb agreement data** for syntactic generalization experiments;
* **SQuAD** for question answering;
* **GloVe embeddings** in the SQuAD experiments.

Dataset files and pretrained embeddings are not redistributed in this repository. Before running an experiment, inspect the first cells of the corresponding notebook for its expected file paths and download instructions.

A future archival release will provide unified data-preparation commands and checksum-verified dataset instructions.

## Reproducing an experiment

To compare faster query/key learning with the baseline:

1. Select a dataset or controlled-experiment directory.
2. Run the baseline notebook from a fresh kernel.
3. Record its random seed, learning rates, model configuration, and final metrics.
4. Run the corresponding faster-QK notebook under the same configuration.
5. Compare both task performance and attention-based measurements.



## Citation

<!--- Please cite the paper when using this code:

```bibtex
@inproceedings{AUTHOR_KEY_2026_faster_qk,
  title     = {Faster Query-Key Learning Sharpens Attention in Self-Attention Models},
  author    = {AUTHOR_NAMES},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```-->

The proceedings citation will be added when the public ICML record becomes available.

For research-related questions, contact: **rahul@cse.iitm.ac.in**
