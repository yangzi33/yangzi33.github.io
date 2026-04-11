---
layout: post
title: "The Calculus of Self-Correction: Why Agentic Loops are Dynamical Systems"
date: 2026-04-11
---

In the research landscape of 2026, we’ve moved past the novelty of "Chain of Thought." We are now in the era of **Agentic Loops**—systems that don't just output a string of text, but iterate through a cycle of planning, acting, and reflecting. 

But if you look under the hood of most current agents, the "Reflection" step is a bit of a mathematical black box. We’re essentially asking an LLM to grade its own homework. If we want these agents to be reliable in high-stakes environments, we need to stop treating reflection as a prompt and start treating it as a **Control Problem.**

---

## The Latent State of a Plan

Let’s formalize the "thought process." When an agent is working on a task, its current plan isn't just a string; it’s a point $z_t$ in a high-dimensional latent space $\mathcal{Z}$. 

Every time the agent observes the environment ($o_t$) and "reflects" on its progress, it applies a **Reflection Operator** $R$. We can model the evolution of the agent’s logic as a discrete-time dynamical system:

$$z_{t+1} = R(z_t, o_t; \theta)$$

The burning question is: **Does this sequence $\{z_t\}$ converge?** Or is the agent just spiraling into a "hallucination sink"?

---

## Convergence via Contraction Mappings

If we want an agent to actually finish a task, the operator $R$ needs to behave. Specifically, we want $R$ to be a **Contraction Mapping**. 

In metric space terms, if we have two different potential plans $z_a$ and $z_b$, applying the reflection operator should bring them closer together. Holding the current observation $o$ fixed and treating $R(\cdot, o; \theta)$ as a map on $\mathcal{Z}$, we’re looking for a Lipschitz constant $k < 1$ such that:

$$d(R(z_a, o;\theta), R(z_b, o;\theta)) \leq k \cdot d(z_a, z_b)$$

When this condition is met on a complete metric space, the **Banach Fixed-Point Theorem** guarantees that the sequence $\{z_t\}$ converges to a unique fixed point $z^{\ast}$—regardless of the initial state. Here $z^{\ast}$ is the latent representation of a *goal-satisfying plan*: the state where the agent has nothing left to revise, i.e., $R(z^{\ast}, o; \theta) = z^{\ast}$. Crucially, this only guarantees *convergence to* $z^{\ast}$, not that $z^{\ast}$ is correct—whether the fixed point actually represents a good plan is a design requirement on $R$, not a consequence of the theorem. If $k = 1$ the map is non-expansive and convergence is no longer guaranteed; if $k > 1$ the plans can actively diverge, producing the "overthinking into nonsense" failure mode we see in unoptimized agentic loops.



---

## Lyapunov Stability: Defining the "Goal Energy"

How do we prove an agent won't diverge? We can borrow a page from classical control theory and define a **Lyapunov Function** $V(z)$, which acts as an "energy" or "error" metric. 

For a stable agent, $V$ must satisfy two conditions: it must be non-negative and zero only at $z^{\ast}$ — the goal-satisfying fixed point defined above ($V(z) \geq 0$, $V(z^{\ast}) = 0$) — and each step of reflection must decrease it in expectation:

$$\mathbb{E}[V(z_{t+1}) - V(z_t) \mid z_t] < 0$$

This gives us a rigorous way to evaluate "Memory Hygiene." When an agent "forgets" irrelevant context, it is effectively performing a **Projection Operator** $P$ that maps a messy, high-entropy state back onto a lower-dimensional manifold of "feasible plans." 

By pruning the history, we aren't just saving tokens; we are forcing the state $z_t$ to stay within a compact region where $V(z)$ is well-behaved.

---

## The Shift: From Prompts to Operators

The "Prompt Engineering" era was about finding the right words. The "Agentic Theory" era is about finding the right **operators**. 

If we can design reflection mechanisms that are mathematically guaranteed to be contractive, we move from "it usually works" to "it is theoretically bound to converge." This is the bridge between the fluid intuition of Large Language Models and the hard guarantees of optimization theory.

The next time you see an agent "thinking," don't just look at the tokens. Look at the trajectory in $\mathcal{Z}$. The math of the loop is where the real intelligence lives.
