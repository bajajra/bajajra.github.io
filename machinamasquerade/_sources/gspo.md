# GSPO: Group Sequence Policy Optimization

GSPO, introduced by Qwen team at Alibaba. upgrades reward‑aligned RL by shifting the optimisation unit from individual **tokens** to whole **sequences**.  By matching the way we **reward** (sequence‑level judgement) with the way we **update** (sequence‑level gradients), GSPO eliminates length‑dependent variance, trains faster, and is dramatically more stable especially for Mixture‑of‑Experts language models.



> *Use exactly **one** importance‑sampling ratio per answer.*

Compute the log‑probability of the whole answer under the new and old policies, normalise by length, exponentiate, and clip **once**:
$$
[
  s_i(\theta)
  \;=\;
  \exp\!\left(
    \frac{
      \log \pi_{\theta}\!\bigl(y_i \mid x\bigr)\;-\;
      \log \pi_{\theta_{\text{old}}}\!\bigl(y_i \mid x\bigr)
    }{%
      \lvert y_i\rvert
    }
  \right)
]
$$
All tokens in that answer now share the same weight.  The variance term that scaled with sequence length disappears.

---

##   From PPO to GRPO : what broke?

**Proximal Policy Optimization (PPO)** is still the work‑horse of RLHF, but it needs an auxiliary value head that doubles memory and often diverges on long texts.

**Group Relative Policy Optimization (GRPO)** removes the value head by sampling *G* responses for the same prompt, ranking them, and treating the z‑scored reward as an *advantage*.  
The catch: GRPO applies an importance‑sampling ratio **per token**.  On a thousand‑token answer, you multiply a thousand noisy numbers due to which variance grows with length, and gradients become unstable.  Sparse MoE models are particularly vulnerable: tiny token‑level swings change expert routing and can collapse the network.

---

##   GSPO v/s GRPO

| Pain‑point in GRPO | GSPO’s fix                            |
| --- |---------------------------------------|
| Reward is per **sequence** but gradient per **token** | Aligns units: both are sequence‑level |
| Variance grows linearly with length | Single ratio ⇒ variance ≈ constant    |
| MoE “expert flipping” under noise | Token noise is averaged away          |
| Need for routing‑replay caches | No longer required                    |

---

##   Algorithm 
1. **Roll‑out:** for each prompt, sample *G* answers with a frozen policy.  
2. **Reward:** score each answer (human or model).  
3. **Advantage:** z‑score within the group.  
4. **Ratio:** compute sequence‑level importance weight.  
5. **Clip:** apply a single clip range (ε≈0.2–0.5).  
6. **Back‑prop:** every token gets the same scalar weight.  
7. **Sync:** periodically update the frozen policy.

### PyTorch skeleton

```python
def gspo_step(model, old_model, prompts, responses, rewards,
              clip_eps=0.4, optimizer=None):
    """
    model      : current policy (nn.Module)
    old_model  : frozen policy used for roll-outs
    prompts    : list[str]  (N*G)
    responses  : list[str]
    rewards    : tensor(N*G)  scalar reward
    """
    # reshape rewards to (N, G)
    B = len(prompts) // 4   # assume G=4 for simplicity here
    rewards = rewards.view(B, 4)

    # group‑normalised advantages
    mu  = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
    adv = ((rewards - mu) / std).view(-1)

    # sequence log‑prob under both policies
    with torch.no_grad():
        lp_old = old_model.loglikelihood(prompts, responses)  # (N*G,)
    lp_new = model.loglikelihood(prompts, responses)          # (N*G,)

    lens = torch.tensor([len(r.split()) for r in responses],
                        device=lp_new.device)
    s = torch.exp((lp_new - lp_old) / lens)

    w = torch.minimum(s * adv,
                      torch.clamp(s, 1 - clip_eps, 1 + clip_eps) * adv)

    # token‑level log‑probabilities
    token_logp = model.token_logprobs(prompts, responses)
    loss = -(w.repeat_interleave(lens) * token_logp).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

##   Empirical Highlights and Knobs

| Hyper‑parameter | Typical GSPO range | Reason |
| --- | --- | --- |
| **Group size G** | 2–8 (4 sweet‑spot) | Sharper relative reward without VRAM blow‑up |
| **Clip ε** | 0.2–0.5 | Sequence ratios are tighter; wider clip prevents under‑training |
| **Learning rate** | 5 e‑6 – 2 e‑5 | Variance is lower, allowing higher LR |
| **Batch size** | 2 000–4 000 sequences | GSPO robust to stale batches; decouple roll‑out & update |
| **MoE routing replay** | Disable | Sequence weight already stabilises experts |


* **Training speed:** Qwen‑3 30B hits the same math benchmark **1.6 × faster** than GRPO.  
* **Stability:** 500 K RL steps on 64‑expert MoE models show **zero collapses** (vs several with GRPO).  
* **Throughput:** Fewer forward passes and no routing caches reduce GPU memory by ~10 %.

---

##   Cases where it might not help

* Tasks with **token‑level rewards** (e.g. real‑time ASR).  
* **Ultra‑short** outputs (< 5 tokens) where variance gains vanish.  

---

## Hugging Face TRL Quick‑start (GSPO)

Using the exact same GRPO script along with `importance_sampling_level` flag set as **sequence** flips the training algo to GSPO.

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("SmallDoge/SmallThoughts", split="train")

def reward_func(completions, **kwargs):
    pass

gspo_cfg = GRPOConfig(
    output_dir="Qwen2-0.5B-GSPO",
    importance_sampling_level="sequence",   # the GSPO switch
    epsilon=0.4,
    group_size=4,
    learning_rate=1e-5,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=gspo_cfg,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()
```

---