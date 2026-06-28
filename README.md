# Faster Query–Key Learning Sharpens Attention in Self-Attention Models

**Official code repository for our ICML 2026 paper.**

[Paper](https://openreview.net/pdf?id=yESGs2JgSu) · [Project page](https://github.com/vashishtrahul/Faster-Query-Key-Learning-Sharpens-Attention-in-Self-Attention-Models)

## Overview

Self-attention contains two interacting circuits with distinct functional roles:

* the **query–key (QK) circuit** governs attention allocation,
* the **output–value (OV) circuit** maps attended representations to predictions.

This work studies a optimization intervention, both theoretically and empirically:

> **What happens when the query–key circuit learns faster than the output–value circuit?**

We show that increasing the learning rate of the QK circuit relative to the OV circuit causes self-attention models to place more probability mass on task-relevant tokens. In other words, **faster query–key learning produces sharper and more selective attention**.

We study this phenomenon through:

* a theoretical analysis of gradient-flow dynamics in a tractable self-attention model;
* controlled synthetic experiments;
* fixed-attention experiments that isolate the effect of attention sharpness;
* experiments on HateXplain, subject–verb agreement, and SQuAD.

## Main idea

Let <strong>η<sub>QK</sub></strong> denote the learning rate of the query–key circuit and <strong>η<sub>OV</sub></strong> denote the learning rate of the output–value circuit.

The standard setting uses equal learning rates:

**η<sub>QK</sub> = η<sub>OV</sub>**

We study faster-QK training, where:

**η<sub>QK</sub> > η<sub>OV</sub>**

The key control parameter is therefore the learning-rate ratio:

**r = η<sub>QK</sub> / η<sub>OV</sub>**

Larger values of **r** allocate relatively more optimization progress to learning *where to attend*, rather than only increasing the strength of the output prediction.

## Theoretical results

We analyze a stylized next-token prediction problem in which the gradient-flow dynamics of self-attention reduce to two scalar quantities:

* **μ<sub>QK</sub>(t):** the scalar parameter governing the query–key circuit and, consequently, the concentration of attention on relevant tokens.
  
* **μ<sub>OV</sub>(t):** the scalar parameter governing the output–value circuit and its contribution to the prediction scores.


The total attention mass placed on informative tokens is:

```text
α(t) = m exp(μ_QK(t)) / [m exp(μ_QK(t)) + n]
```

Thus, increasing <strong>μ<sub>QK</sub></strong> directly increases the model’s attention to relevant tokens.

The prediction loss depends on both circuits through their product:

```text
Loss(t) = log[1 + (M − 1) exp(−α(t) μ_OV(t))]
```

This expression captures an important trade-off: the model can improve its prediction either by increasing the OV strength or by sharpening its attention so that a larger fraction of the OV signal is applied to informative tokens.

### Main theoretical conclusions

1. **The QK and OV circuits follow coupled learning dynamics.**
   The OV circuit begins learning the prediction signal, while its growing strength subsequently drives the learning of the QK circuit.

2. **Training progressively sharpens attention.**
   As <strong>μ<sub>QK</sub></strong> increases, <strong>α(t)</strong> increases and the model places more attention mass on relevant tokens.

3. **The relative learning rate determines how the model reaches low loss.**
   At a fixed loss level, a larger value of <strong>η<sub>QK</sub> / η<sub>OV</sub></strong> produces a stronger QK circuit and sharper attention, while requiring less growth in the OV circuit.

4. **Faster QK learning changes the learned solution, not merely the training speed.**
   Models can reach comparable prediction loss through different combinations of attention sharpness and OV magnitude. Increasing the QK learning rate biases training toward solutions that rely more strongly on selective attention.

5. **Attention becomes concentrated on informative tokens during continued training.**
   In the theoretical dynamics, the informative-token attention mass approaches one, while the rate and finite-time degree of concentration depend on the QK-to-OV learning-rate ratio.

These results provide a mechanistic explanation for the empirical observation that faster query–key learning sharpens attention.

<details>
<summary><strong>Reduced gradient-flow equations</strong></summary>

For the theoretical model, the circuit strengths evolve according to:

```text
dμ_OV/dt =
η_OV α
────────────────────────────────
b [exp(α μ_OV) + M − 1]
```

and

```text
dμ_QK/dt =
η_QK (M − 1) α(1 − α) μ_OV
────────────────────────────────
M b² [exp(α μ_OV) + M − 1]
```

These equations make the coupling explicit:

* QK learning is proportional to the current OV strength;
* QK learning is strongest when attention is neither completely diffuse nor completely saturated;
* increasing <strong>η<sub>QK</sub></strong> directly accelerates the growth of the attention-selecting circuit.

The simulations in `Fig-4 Code/Simulated_Differential_equations.ipynb` numerically integrate these equations across multiple values of <strong>η<sub>QK</sub> / η<sub>OV</sub></strong>.

</details>

## Empirical findings

The experiments test the theoretical prediction across controlled and natural-language settings. They examine whether faster QK learning:

* increases attention on informative or task-relevant tokens;
* produces sharper attention distributions;
* changes the balance between QK and OV circuit learning;
* improves or preserves downstream task performance;
* improves attention-based explanation metrics on datasets with token-level rationales.

Together, the theory and experiments support a consistent conclusion:

> **The relative optimization speed of attention’s internal circuits is an important inductive bias. Faster query–key learning encourages models to solve tasks using sharper, more selective attention.**






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
