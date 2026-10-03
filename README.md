# MAGE

**MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs**

**Accepted at NeurIPS 2026**

Ruiyi Yang, Zechen Li, Hao Xue, Imran Razzak, and Flora D. Salim

[Paper](https://arxiv.org/abs/2605.10064) · [PDF](https://arxiv.org/pdf/2605.10064) · [Code](https://github.com/RuiyiYang01/mage-alg)

## What MAGE does

MAGE (Multi-Agent Graph-guided Evolution) improves a frozen language-model agent by evolving its external knowledge and decision policies. EvoKG connects four kinds of knowledge: **capabilities, tasks, experiences, and environments**. A stronger teacher supplies corrections for failures, while the learner contributes its own successful reasoning traces. The learner retrieves relevant experience to guide subsequent tasks.

Multiple agents coordinate skill discovery, curriculum selection, exploration, evaluation, and graph updates. Task-level search and skill-level strategy bandits adapt from the reward stream alongside the graph. The paper studies this process across nine benchmarks covering reasoning, question answering, spatiotemporal tasks, finance, medicine, survival, and web navigation. See the [paper](https://arxiv.org/abs/2605.10064) for the analysis and evaluation.

## Repository contents

| Path | Purpose |
| --- | --- |
| `mage/core/` | EvoKG schemas, graph operations, evolution loops, and bandit policies |
| `mage/agents/` | Discovery, navigator, explorer, learner, critic, and curator |
| `mage/environments/` | Reasoning and interactive environment adapters, plus a toy grid world |
| `mage/tools/` | Tool registry, computation tools, retrieval, and tool execution |
| `mage/utils/` | LLM backends, embedding retrieval, and answer judging |
| `mage/training/` | Optional SFT/GRPO implementations used by training-based learner variants |
| `configs/` | Editable configurations for nine benchmark environments |
| `scripts/` | Toy demo, main evolution entry point, and held-out evaluation |
| `tests/` | Graph and experience-bandit unit tests |

The paper's frozen-learner setup uses `learner.type: routed`. The package also retains the ICL and training-based learner implementations. Configuration headers identify their original source configurations; output names and local endpoints have been normalized for this release.

This repository contains source code, configuration, and documentation. Datasets, experiment results, model weights, logs, saved graphs, generated tools, and retrieval caches are excluded. Running experiments creates local artifacts under the ignored `experiments/` directory; generated tools and Wikipedia caches are also ignored.

## Installation

Use Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/RuiyiYang01/mage-alg.git
cd mage-alg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[llm,dev]'
```

The `llm` extra provides API clients, dataset loading, and embedding retrieval. Optional extras are `crafter` for the Crafter adapter, `spatial` for spatial computation, and `quantization` for local 4-bit model loading. Local model inference requires hardware suitable for the chosen model; an OpenAI-compatible inference server can host the learner separately.

## Quick start: no API or dataset required

```bash
python scripts/run_toy_evolution.py 3
```

This runs three iterations with stub agents in a toy grid world, exercising the graph and evolution loop. It is a functional demo; benchmark runs use the main entry point below.

## Run MAGE

1. Set the teacher credential:

   ```bash
   export ANTHROPIC_API_KEY='your-api-key'
   ```

2. Start your learner's OpenAI-compatible inference service. Edit `llm.weak.model` and `llm.weak.base_url` in the selected YAML to match that service. The example endpoint is `http://localhost:8000/v1`. Set `OPENAI_API_KEY` if the endpoint requires authentication. Set `llm.strong.model` to the teacher model you intend to use; the supplied IDs reflect the source configurations.

3. Run evolution, for example on GSM8K:

   ```bash
   python scripts/run_evolution_v2.py --config configs/gsm8k.yaml --iterations 10 --seed 42
   ```

The configuration controls the environment, data split, task budget, learner routing and memory options, curriculum, and number of iterations. `experiment.output_dir` determines where the run stores its graph, policies, and logs. Re-running a directory can reuse discovery and generated state; use a fresh output directory for an independent run.

### Benchmark setup

| Config | Data and environment setup |
| --- | --- |
| `gsm8k.yaml` | GSM8K; loaded through Hugging Face Datasets |
| `hotpotqa.yaml` | HotpotQA; loaded through Hugging Face Datasets |
| `webqa.yaml` | WebQuestions; loaded through Hugging Face Datasets |
| `realmath.yaml` | RealMath; loaded through Hugging Face Datasets |
| `finqa.yaml` | FinQA; loaded through Hugging Face Datasets |
| `medqa.yaml` | MedQA; loaded through Hugging Face Datasets |
| `stbench.yaml` | Supply benchmark split files in `environment.data_dir` (default: `data/stbench/splits`) |
| `crafter.yaml` | Install `pip install -e '.[crafter]'` |
| `webshop.yaml` | Install WebShop with its data and search dependencies; set `WEBSHOP_DIR` to its checkout |

For the optional ST-Reasoner computation pipeline, set `ST_REASONER_DIR` to a checkout exposing `tools.spatial_ops`, `tools.temporal_ops`, `tools.graph_ops`, and `tools.stats_ops`. These external tools and benchmark datasets are obtained separately. The built-in MAGE tool registry remains in `mage/tools/st_tools.py`.

Some configurations enable retrieval from a teacher rationale pool. Such pools are generated artifacts and are excluded; the retrieval implementation falls back to training examples when a pool is absent. Consult the paper and configure the same data, models, and retrieval resources when reproducing a specific setting.

### Held-out evaluation

`scripts/eval_held_out.py` supports GSM8K, HotpotQA, WebQuestions, RealMath, and STBench. Reserve examples by setting `environment.max_instances` and `environment.held_out_size` before evolution. Their sum must fit the selected dataset split. Use the same configuration and seed for evolution and evaluation, with `--num-test` matching the reserved size.

For example, set `max_instances: 1000` and `held_out_size: 100` in `configs/gsm8k.yaml`, run evolution into a fresh output directory, then evaluate:

```bash
python scripts/eval_held_out.py \
  --config configs/gsm8k.yaml \
  --experiment experiments/gsm8k \
  --num-test 100
```

The evaluator loads the latest saved iteration by default; `--iter N` selects one explicitly.

## Tests

```bash
python -m pytest tests -q
```

## Citation

If you use MAGE, please cite the paper. The BibTeX below identifies the arXiv version; the paper has been accepted at NeurIPS 2026.

```bibtex
@misc{yang2026mage,
  title         = {{MAGE}: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs},
  author        = {Ruiyi Yang and Zechen Li and Hao Xue and Imran Razzak and Flora D. Salim},
  year          = {2026},
  eprint        = {2605.10064},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2605.10064},
  url           = {https://arxiv.org/abs/2605.10064},
  note          = {Accepted at NeurIPS 2026}
}
```
