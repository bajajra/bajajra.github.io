# GRPO: Huggingface Implementation
## GRPO in a Nutshell  

GRPO maximises the *advantage* of each sampled completion without a learned critic.  
For a prompt \(x\) we draw a **group** of \(G\) completions $\{y_i\}$, compute task reward $r_i$ for each, and *z-score* normalise  

$$
\begin{aligned}
\tilde r_i &= \frac{r_i - \mu_r}{\sigma_r}, \\[6pt]
\mu_r      &= \frac{1}{G}\sum_{j=1}^{G} r_j, \\[6pt]
\sigma_r   &= \sqrt{\frac{1}{G}\sum_{j=1}^{G} \bigl(r_j - \mu_r\bigr)^2}.
\end{aligned}
$$
The objective is

$$
\mathcal{L}(\theta)=
\sum_{i=1}^{G} \min\!\Bigl(
r_{i,t}(\theta)\,\tilde r_i,\,
\operatorname{clip}\bigl(r_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\tilde r_i
\Bigr)
-\beta\,D_{\mathrm{KL}}\bigl(\pi_{\theta}\,\Vert\,\pi_{\text{ref}}\bigr).
$$

where  
$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(y_i \mid x)}{\pi_{\text{ref}}(y_i \mid x)}.
$$
Because there is **no critic head**, you save 30–40 % GPU memory and can scale to larger models.

---

## Environment & Dependencies  

> **Tip :** use uv for package and environment management


```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

#create and activate virtualenv
uv venv --seed --python==3.12
source .venv/bin/activate

# core libraries
pip install "trl[vllm]==0.16.1" transformers datasets accelerate peft bitsandbytes hf-tranfer

# (optional) Flash‑Attention for extra speed
pip install flash-attn==2.*
```

> **Tip :** install Flash‑Attention *before* Transformers to avoid rebuilding kernels.

---

## Dataset Preparation  

`GRPOTrainer` expects an **untokenised** `datasets.Dataset` with two mandatory columns:

| Column     | Type | Purpose                                 |
|------------|------|-----------------------------------------|
| `prompt`   | str  | user / system input                     |
| `response` | str  | reference answer (optional)             |

Any extra fields (e.g. `ground_truth`, `score`) are accessible inside a custom reward function.

```python
from datasets import load_dataset

ds = load_dataset("trl-lib/tldr", split="train")

# keep first 50 000 rows for a demo run
ds = ds.shuffle(seed=42).select(range(50_000))
```

---

## Designing a Reward Function  

A **reward callback** receives a batch of `{prompt, completion, …}` dictionaries and returns an array of floats.  
Below is a minimal example that penalises long outputs and rewards well‑formed sentences:

```python
import numpy as np
from nltk.tokenize import word_tokenize

def simple_reward(samples):
    out = []
    for s in samples:
        length_penalty = -0.001 * max(0, len(word_tokenize(s["completion"])) - 512)
        end_bonus      = 1.0 if s["completion"].strip().endswith(".") else 0.0
        out.append(length_penalty + end_bonus)
    return np.array(out, dtype=np.float32)
```

> **Normalise** raw rewards to roughly $\\([-5,5]\\)$ **before** the z‑score step; extreme values slow convergence.

---

## Wiring `GRPOTrainer`  

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,      # quantised weights
    device_map="auto")      # spread over available GPUs

config = GRPOConfig(
    group_size                = 4,      # completions per prompt
    beta                      = 0.05,   # KL‑penalty factor
    learning_rate             = 5e-6,
    mini_batch_size           = 4,
    batch_size                = 128,
    target_kl                 = 0.1,    # auto‑tune beta
    gradient_accumulation_steps = 4,
    reward_fn                 = simple_reward,
)

trainer = GRPOTrainer(
    model         = model,
    ref_model     = None,       # frozen copy instantiated internally
    args          = config,
    tokenizer     = tokenizer,
    train_dataset = ds,
)

trainer.train()
```

Key points:

* **`ref_model=None`** duplicates the initial weights internally—saves RAM.  
* `group_size = 4–8` is a sweet spot: lower variance without exploding memory.  
* If **actual** KL rises above `target_kl`, the trainer automatically scales `beta`.

---

## Advanced Configuration & Scaling  

### LoRA + 4‑bit Quant  

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
```

Reduces a 7B model from ~28 GB to < 9 GB VRAM.

---

### Multi‑GPU & FSDP  

```bash
CUDA_AVAILABLE_DEVICES=<> accelerate launch --num_processes <gpu_devices> --mixed_precision fp16 grpo_run.py
```

---

## Choosing Hyper‑parameters  

| Hyper‑parameter     | Typical Range | Intuition                                             |
|---------------------|---------------|-------------------------------------------------------|
| `group_size`        | 4 – 8         | bigger → lower variance; but O(G) RAM                 |
| `beta`              | 0.01 – 0.1    | KL penalty strength; auto‑tuned if `target_kl` set    |
| `learning_rate`     | 2e‑6 – 1e‑5   | larger models usually need a smaller LR               |
| `mini_batch_size`   | 1 – 16        | memory‑bound; raise if you have room                  |
| `ppo_epochs`        | 1 – 4         | more passes if data is expensive                      |

> Empirically, `beta = 0.05` and `group_size = 6` maximise reasoning accuracy for 7B models.


#### Track:

* **KL divergence** per step – rising trend ⇒ increase `beta`.
* **Reward mean / std** – should hover around 0 after z‑score normalisation.
* **Tokens/sec** – sudden drops often indicate device mapping issues.

---

## Practical Learnings 

| Issue                 | Symptom                              | Remedy                                                                  |
|-----------------------|--------------------------------------|-------------------------------------------------------------------------|
| Trainer **hangs**     | process never exits                  | load without `device_map='auto'` or upgrade `accelerate`                |
| **Slow throughput**   | < 0.2 it/s on small dataset          | increase `mini_batch_size`; ensure `torch.compile` is not disabled      |
| **OOM crash**         | during forward pass                  | enable 4‑bit, LoRA; reduce `group_size`; use FSDP                       |
| **Diverging KL**      | KL > 1.0 quickly                     | raise `beta`, lower LR, narrow `clip_range`, inspect reward scale       |
| **Flat reward signal**| reward variance < 0.5                | enlarge `group_size` or switch to pairwise relative rewards             |

>**Curriculum**: start with `group_size = 8` then decay to 4 – mimics critic annealing.  
>**Reward‑model distillation**: freeze an outperforming checkpoint as the new reference to stabilise KL drift on long runs.

---

## Correlating Theory with Code  

* `group_size` in the config **is** $\\(G\\)$ in the objective.  
* `beta` multiplies the KL term.  
* Your `reward_fn` provides $\\(r_i\\)$.  
* TRL normalises rewards internally:

  ```python
  advantage = (r - r.mean()) / (r.std() + 1e-8)
  ```

  matching the $\\(\\tilde r_i\\)$ definition.  
* The `min` + `clip` logic is executed in a fused CUDA kernel for speed.

---

