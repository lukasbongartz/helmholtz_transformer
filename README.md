## Quickstart

Run the experiments that generate entropy/free-energy plots using DistilGPT2.

### Setup

```bash
cd "/Users/lukasbongartz/Desktop/Transformer Paper/code/helmholtz_transformer"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run

```bash
python run_hfe_experiments.py \
  --model distilgpt2 \
  --dtype float32 \
  --device cpu \
  --dt 1.0 \
  --max_length 64 \
  --save_dir hfe_out
```

Outputs (plots and a summary `.pt`) are written to `hfe_out/`.

### Notes

- The script downloads the model and tokenizer from the Hugging Face Hub on first run.
- GPU is optional; set `--device cuda` if available.
```
# Helmholtz Transformer

Experiments accompanying the paper The Helmholtz Perspective.

See also [ARGΩΣ](https://argos-viz.fly.dev/), a visualisation tool for the latent semantic flow of LLMs.
