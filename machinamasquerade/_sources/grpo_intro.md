# Group Relative Policy Optimisation (GRPO): Overview

Group Relative Policy Optimization (GRPO) is a lightweight twist on PPO designed to sharpen an LLM’s reasoning without the heavy critic network classic RL methods rely on. Instead, the model answers each prompt k times, forming a group of candidate outputs. Every answer is scored (for example, by a reward model or a task-specific metric). GRPO treats the group-mean score as the baseline, so the advantage for each answer is simply its score minus that mean—eliminating the need to learn a separate value function and cutting memory and compute costs.

```{figure} images/grpo_process.png
:name: grpo_process
:width: 70%
:align: center

Training Process
```

The policy is then nudged toward higher-than-average answers using a clipped PPO-style ratio to keep updates stable and a KL penalty to stop the new policy from drifting too far from the reference model. Because it works entirely with relative scores inside each mini-batch, GRPO stays sample-efficient, scales to very large models, and has proven effective in real projects, delivering strong gains in mathematical reasoning while using fewer GPUs.

---

## Why another RLHF algorithm?

 - **No critic network required** – the group-average reward is the baseline, cutting memory consumption by ~30 %.
 - **Lower reward variance** – supports higher learning rates and smoother gradients.
 - **Parallel candidate sampling** – keeps GPUs saturated, trimming wall-clock training time by 1.3–1.5 × compared with PPO.
---

## From PPO to GRPO: the mathematical step

### Recap: PPO clip objective

The clipped PPO loss for a single trajectory is

$$
\begin{aligned}
L_{\text{PPO}}(\theta)
  &= \frac{1}{T}\sum_{t=1}^{T}
     \min\!\Bigl(
       r_t(\theta)\,\hat A_t,\,
       \operatorname{clip}\!\bigl(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\hat A_t
     \Bigr), \\[6pt]
r_t(\theta)
  &= \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
\end{aligned}
$$

where $\hat{A}_t$ is the *advantage* predicted by a critic.

### Key idea: ditch the critic

GRPO observes that when you sample **G answers for the *same* prompt**, the raw reward scale is irrelevant; only *which answer is better* matters.  
Define a *group‑normalised* reward

$$
\begin{aligned}
\tilde r_i &= \frac{r_i - \mu_r}{\sigma_r}, \\[6pt]
\mu_r      &= \frac{1}{G} \sum_{j=1}^{G} r_j, \\[6pt]
\sigma_r   &= \sqrt{\frac{1}{G}\sum_{j=1}^{G} \bigl(r_j - \mu_r\bigr)^2}.
\end{aligned}
$$

Set the token‑level advantage of every token in answer $
i \mapsto \tilde{r}_i
$.  
Plugging this into PPO gives the **GRPO objective**

$$
\begin{aligned}
L_{\text{GRPO}}(\theta)
\; &= \sum_{i=1}^{G}\sum_{t}
      \min\!\Bigl(
        r_{i,t}(\theta)\,\tilde r_i,\,
        \operatorname{clip}\!\bigl(r_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\tilde r_i
      \Bigr) \\
    &\quad - \beta\,D_{\mathrm{KL}}\!\bigl(\pi_{\theta}\;\|\;\pi_{\text{ref}}\bigr).
\end{aligned}
$$

### Algorithm in prose

1. **Sample** *G* answers from the current policy for each prompt.  
2. **Score** them with a reward model (RM) or heuristic oracle.  
3. **Normalise** scores within the group → $\tilde r$.  
4. **Back‑prop** through the GRPO loss for μ gradient steps.  
5. **(Optional)** after *K* iterations, refresh the reference model and replay buffer for stability.

No baseline network, no TD‑error, no bootstrapping—just statistics and a clip‑style surrogate.

---

## Toy Example (Arithmetic)

Prompt: “What is 17 × 12?”  
Policy samples *G* = 4 answers:

| i | Answer | Reward (exact‑match) |
|---|--------|----------------------|
| 1 | 204    | +1 |
| 2 | 214    | 0  |
| 3 | 200    | 0  |
| 4 | 240    | 0  |

*Group statistics*: 


$$
\mu = 0.25,\qquad
\sigma \approx 0.43
\;\Longrightarrow\;
\tilde r = \bigl[\, +1.75,\,-0.58,\,-0.58,\,-0.58 \bigr].
$$

Every token of answer 1 gets positive advantage, others negative.  
Gradient ascent on the GRPO objective quickly pushes the logits toward “2 0 4” without ever computing a value function.

---

## PPO v/s GRPO

| Property | PPO | **GRPO** |
|----------|-----|----------|
| Needs critic? | **Yes** (big) | **No** |
| Sample efficiency | Good | Slightly lower (uses groups) |
| Memory | 2 × policy | 1 × policy |
| Hyper‑parameters | ε, β & value‑loss mix | ε, β only |
| Alignment signal | Scalar RM | Same |

Empirically, GRPO yields **fewer degenerate updates** and *better long‑horizon reasoning* because advantages are tied to complete solutions, not per‑token predicted values.

---

