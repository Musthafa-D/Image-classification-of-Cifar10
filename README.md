# CIFAR-10 Image Classification and Model Interpretation

Convolutional image classification on CIFAR-10, with configurable network depth, output-layer comparisons and model interpretation. The project combines training and parameter search with tools for examining classification errors and feature attribution.

The ten categories are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Implementation

- Configurable CNN blocks and network dimensions.
- Linear and NLRL output-head experiments through the existing framework.
- Training and evaluation with loss and classification metrics.
- Optuna-based parameter studies and a separate fixed-configuration launcher.
- Learning-curve, prediction, attribution and representation plots.

The study configuration includes image augmentation and a cross-entropy objective. Record the transforms, architecture and data split when comparing runs.

## Repository guide

| File | Purpose |
|---|---|
| `networks.py` | Network and layer definitions |
| `data_loader.py` | Framework-based dataset preparation |
| `learner.py` | Training, evaluation and experiment recording |
| `optuna_hyp.py` | Hyperparameter study workflow |
| `main.py` | Study launcher using `config.yaml` |
| `dummy_main.py` | Fixed-run launcher using `dummy_config.yaml` |
| `metrics.py` and `plots.py` | Metrics and diagnostic visualisations |
| `StudySummary.ipynb` | Saved study-summary notebook |

The `dummy_*` names refer to the original single-run workflow; they do not imply use of a different synthetic dataset.

## Environment and use

The code uses PyTorch and the external `ccbdl` framework for configuration, data loading, experiment storage and parts of the learning workflow. A compatible installation of that framework is required; it is not bundled here. Other dependencies include torchvision, NumPy, Matplotlib, Optuna and Captum, with additional analysis libraries used by individual modules.

Use the original compatible environment, prepare the dataset at the configured location, and run from the repository root so relative paths resolve correctly. The archive does not include a complete dependency lock file. Supply datasets and optional pretrained models separately where referenced.

Inspect `dummy_config.yaml` and the corresponding learner first. After preparing the data and environment, the fixed-run entry point is:

```bash
python dummy_main.py
```

For a parameter study, use `main.py` with a complete `config.yaml` containing the required study and optimisation settings. Optional plotting/evaluation paths may require specific saved checkpoints.

## Results and interpretation

The code records learning curves, classification metrics and model explanations. Read the configuration and saved run outputs together when reporting a result. This README does not assert a benchmark accuracy or newly reproduced training result.

The original experiments use metrics named `TestAcc` for model selection. These selection scores should not automatically be described as independent final-test performance. Attribution plots inspect the trained model; they do not prove that it reasons like a person.

## Project focus

The work brings together model configuration, parameter search and interpretation. Dataset handling and NLRL/framework components rely on external implementations; distinguish those components from the experiment configuration and analysis in this repository.
