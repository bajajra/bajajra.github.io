# RL Introduction

[//]: # (This chapter explores the role of reinforcement learning &#40;RL&#41; in the development of large language models &#40;LLMs&#41;. It is designed for data scientists looking to understand both the theoretical underpinnings and practical implementations of RL in modern AI systems. We will cover the basics of reinforcement learning, provide a real-world example with self-driving cars, and then delve into how LLMs are trained and aligned using RL principles.)
<!-- YouTube Video Embed at the Top -->
<div align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/I2sN1RmON4I" title="Reinforcement Learning for LLMs: Introduction" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

---

## What is Reinforcement Learning?

Reinforcement learning is a paradigm in machine learning where an **agent** learns to perform tasks through trial and error interactions with its environment. The agent’s objective is to maximize cumulative long-term rewards by continuously adapting its behavior based on received feedback.

### Core Concepts and Notation

In a typical RL problem, the following components are key:

- **Agent** ($A$): The entity that takes actions.
- **State** ($s$): The current situation or environment input.
- **Action** ($a$): The decision or maneuver taken by the agent.
- **Reward** ($r$): The feedback signal provided after an action, which can be positive (reward) or negative (penalty).
- **Policy** ($\pi$): The strategy that maps states to actions.

```{figure} images/rl_process_flow.png
:name: rl_process_flow
:width: 70%
:align: center

RL Setup
```

The interaction typically begins at an initial state $s_0$, where the agent selects an action $a_0$. The environment then returns a reward $r_1$ and transitions to a new state $s_1$.

A key goal in RL is to maximize the cumulative reward, often formalized by the equation:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}
$$

where:
- $G_t$ is the return (cumulative reward) from time step $t$,
- $\gamma$ is the discount factor ($0 \leq \gamma \leq 1$),
- $r_{t+k+1}$ represents the reward received at time $t+k+1$.

---

### Real-World Example: Self-Driving Cars

- **Agent:** The onboard computer system that controls the vehicle.
- **State** ($s$): Sensor data and real-time information about the surrounding environment.
- **Action** ($a$): Possible driving maneuvers, including:
  - Accelerating
  - Braking
  - Steering
  - Changing lanes
  - Coming to a complete stop
- **Reward** ($r$): 
  - **Positive Reward:** Actions that optimize outcomes (e.g., minimizing travel time).
  - **Negative Reward:** Actions that result in poor outcomes (e.g., abrupt braking, collisions).
- **Policy** ($\pi$): The strategy that determines which maneuver to execute in different scenarios based on sensor inputs.


---

## How Large Language Models (LLMs) Operate

LLMs are essentially next-token predictors. They take an input prompt and output the most probable subsequent token based on the learned probability distribution over a vocabulary.

### Next-Token Prediction

Given a sequence of tokens, the LLM calculates the probability for each token in the vocabulary:

$$
P\bigl(t_i \mid t_1, t_2, \ldots, t_{i-1}\bigr)
$$

For example, when the prompt is “president of the USA is”, the model evaluates the probability of each possible continuation token based on its training data.

[//]: # (> **Image:** [LLM Next-Token Prediction]&#40;https://example.com/llm_next_token.png&#41;)

---

## Training Flow for Modern LLMs

The process for training today’s LLMs generally follows these steps:

1. **Pre-training:**
   - The model is trained on a vast amount of text data to learn general language representations.
2. **Supervised Fine-Tuning:**
   - The model is further refined to follow specific instructions and adhere to language structures.
3. **Reinforcement Learning Alignment (RL Alignment):**
   - The model is aligned to meet criteria such as helpfulness, factuality, and harmlessness. This involves training with a reward model that evaluates the quality of responses.

```{figure} images/llm_training_seq.png
:name: llm_training_seq
:width: 100%
:align: center

  LLM Training Flow
```

### Reinforcement Learning Alignment in LLMs

In RL alignment, the following principles guide the LLM’s behavior:

- **Helpfulness:** The LLM should provide comprehensive and useful answers.
- **Accuracy:** It should minimize hallucinations and offer correct, verifiable information.
- **Harmlessness:** The LLM must ensure its outputs are politically correct and avoid disseminating harmful or misinformative content.

Just as a self-driving car follows a series of actions to reach its destination, the RL framework in LLMs involves a trajectory of token generation:

- **Agent:** The LLM.
- **State:** The combination of user input and tokens generated so far.
- **Action:** The generation of the next token.
- **Reward:** An evaluation function that assesses the generated response.
- **Policy:** The probability distribution over tokens that the LLM learns to optimize for cumulative reward.

## Mathematical Perspective on RL in LLMs

To formalize the learning process, consider the following:
- Let $\pi_{\theta}(a \mid s)$ be the parameterized policy (i.e., the token probability distribution) where $\theta$ represents the learnable parameters.
- The objective is to maximize the expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t \, r_t \right]
$$


Optimization methods such as policy gradient techniques can be applied to adjust $\theta$ such that the cumulative reward is maximized.

---


[//]: # (*End of Chapter*)