# grapHIC: Graph-Transformer based Hi-C phasing

![grapHIC Logo](graphic_logo.png)

grapHIC is a deep learning tool for reference-free haplotype phasing that incorporates Hi-C information. It works directly on unitig graphs, which are compressed graph representations of sequencing reads used as a starting point for genome assembly.

## Overview

You can train a model or use our trained model to predict haplotypes in a unitig graph. 
Given a set of hifi and HiC reads, you can run our pipeline to get the following outputs:
- A unitig graph (from Hifiasm)
- HiC contact information (from HiC-Pro)
- Unitig Haplotype predictions (from grapHICs model)

The haplotype predictions then can be used to cluster the reads and seperate haplotypes. If you are working on an assembly tool you can use them to help guiding the further assembly process. If you are interested in just seperating unitigs into haplotypes you can seperate them as we show in eval_main.py.

## Installation

git clone https://github.com/MartinSchmitz95/grapHIC.git
cd grapHIC
pip install -r requirements.txt

## Data

The dataset for grapHIC is available at this [Google Drive link](https://docs.google.com/document/d/1dgaDoSTPlwgjVafBDe3Mq3NbJqoQy-G7vrdMBa4lrZM/edit?usp=sharing).

## Model

Our best model is available in the repository under `trained_models/graphic_model.pt`.
## Usage

### Training

To train a model using grapHIC, use the `train_diploid.py` script:

```python train_diploid.py --data_path /path/to/pyg_graphs/ \
                        --data_config dataset_config.yml \
                        --device cuda:0 \
                        --run_name experiment_name \
                        --mode pairloss_global \
                        --wandb project_name
```

### Evaluation

To evaluate a trained model, use the `eval_main.py` script:

```
python eval_main.py --mode pred \
                   --model_path trained_models/graphic_model.pt \
                   --graph_path /path/to/graph.pt
```

#### Evaluation Parameters
- `--model_path`: Path to trained model
- `--graph_path`: Path to graph file for evaluation
- `--output_dir`: Output directory for results
- `--mode`: Evaluation mode (pred, emb, spectral, louvain, lp)
- `--config_path`: Path to configuration file

## Objectives

grapHIC implements several loss functions:
- `pairloss_global`: Global supervised contrastive pair loss
- `pairloss_local`: Local supervised contrastive pair loss
- `triplet_loss`: Triplet loss for node embeddings
- `subcon`, `info_nce`, `full_cont`: Various contrastive embedding losses

## Dataset Configuration

The dataset configuration file (YAML) specifies how to split your data into training and validation sets.

## Model Architecture

grapHIC uses a graph transformer architecture (SGFormer) that can efficiently process large, heterogeneous graphs by combining:
- Transformer layers for capturing global context
- Graph neural network layers for local message passing
- Support for heterogeneous edge types (overlap and Hi-C connections)
