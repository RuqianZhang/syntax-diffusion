# Syntax-Guided Diffusion Language Model

Research code for syntax-aware diffusion text generation experiments. The repository contains cascade and non-cascade variants, with and without text-to-text conditioning.

## Repository Layout

```text
cascade/          Cascade syntax-to-text model.
cascade-t2t/      Cascade model with text-to-text conditioning.
noncascade/       Non-cascade syntax/text diffusion model.
noncascade-t2t/   Non-cascade model with text-to-text conditioning.
datasets/         Local dataset files and syntax tokenizer metadata.
```

Each experiment folder is self-contained and includes:

```text
train_*.py        Training and sampling entry point.
dataset_utils/    Dataset loading, preprocessing, and collators.
diffusion/        Diffusion process and trainer.
model/            Transformer backbone.
evaluation/       Generation metrics.
scripts/          Example training and generation commands.
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install the CUDA-compatible PyTorch build for your machine if the default `torch` package is not appropriate.

## Data

By default, the code reads datasets from the repository-level `datasets/` directory. You can override this with:

```bash
export SYNTAX_DIFFUSION_DATA_DIR=/path/to/datasets
```

Expected dataset subdirectories:

```text
datasets/
  emotion/{train_data.json,valid_data.json}
  yelp/{train_data.json,valid_data.json}
  syntax-pos.json
```

Classifier checkpoints used by evaluation are read from `./saved_text_classifier` by default. Override with:

```bash
export SYNTAX_DIFFUSION_CLASSIFIER_DIR=/path/to/saved_text_classifier
```

## Running Experiments

Run scripts from any location; they change into their experiment directory automatically.

```bash
bash noncascade/scripts/yelp/train.sh
bash noncascade/scripts/yelp/gen.sh

bash cascade/scripts/yelp/train_syn.sh
bash cascade/scripts/yelp/train_syn2text.sh
bash cascade/scripts/yelp/gen_cascade.sh
```

Generation scripts expect checkpoints under `./ckpts` by default. Override checkpoint roots with:

```bash
export SYNTAX_DIFFUSION_CKPT_DIR=/path/to/ckpts
```

You can also run entry points directly, for example:

```bash
cd noncascade
python train_noncascade.py --dataset_name yelp --self_condition --scale_shift \
  --num_train_steps 250000 --max_seq_len 100 --sampler ddim \
  --sampling_timesteps 250 --train_batch_size 128 --eval_batch_size 128 \
  --num_dense_connections 3 --tx_dim 768 --tx_depth 12 \
  --class_conditional --num_classes 3
```
