# Optimizers as Human Learners

Understanding gradient descent through the lens of how humans learn.

---

## SGD: The Stubborn Learner

**Algorithm:**
```python
w = w - lr * grad
```

**Learner Analogy**

Imagine a student who studies with unwavering consistency. Every mistake gets the same treatment—no matter if it's a small slip or a fundamental misunderstanding. This student sets a fixed study pace and sticks to it, treating calculus homework the same way they treat simple addition/subtraction.

That's Stochastic Gradient Descent.

**How It Works**

SGD updates every weight using the same learning rate (`lr`). It looks at the gradient (the direction of the mistake) and takes a step in the opposite direction. Simple. Reliable. Stubborn.

**Variable Mapping:**
- `w` → Your current understanding of a concept
- `lr` → How fast you learn (your study speed)
- `grad` → The direction and size of your mistake

**What SGD Does Well**

SGD shines in simple, well-behaved problems where the path to the solution is smooth and convex. It's guaranteed to converge if you set the learning rate right. It's also incredibly memory-efficient—no extra state to track, no momentum vectors, no adaptive rates. Just weights and gradients.

For small datasets or problems with clear structure, SGD is often all you need.

**Where SGD Struggles**

The stubbornness becomes a liability in complex terrain. SGD gets stuck in local minima easily because it has no memory of where it's been. In flat regions (plateaus), it crawls painfully slow. And perhaps most critically, it uses the same learning rate for all parameters—some weights might need bold steps while others need careful nudges, but SGD treats them all the same.

**Best Used For:** Convex optimization problems, simple neural networks, when you want guaranteed convergence and have time to tune the learning rate carefully.

---

## Momentum: The Persistent Learner

**Algorithm:**
```python
v = β * v - lr * grad
w = w + v
```

**Learner Analogy**

Think of a student who builds confidence from repeated successes. When they solve several calculus problems correctly in a row, they approach the next one with momentum—faster, more confident. If they hit a tricky problem, that accumulated confidence helps them push through instead of getting stuck.

That's Momentum SGD.

**How It Works**

Momentum maintains a velocity vector (`v`) that accumulates gradients over time. The `β` parameter (typically 0.9) controls how much of the past velocity to retain. Each update combines 90% of the previous momentum with 10% of the current gradient.

This creates acceleration in consistent directions and dampening in oscillating ones.

**Variable Mapping:**
- `v` → Accumulated confidence from past learning
- `β` → How much you remember from previous study sessions (memory retention rate)
- `lr` → Base learning speed
- `grad` → Current mistake

**What Momentum Does Well**

Momentum excels at escaping local minima—if you're stuck in a shallow valley, accumulated velocity can carry you over the edge to find deeper solutions. It converges faster than vanilla SGD because it accelerates in directions where gradients consistently point the same way.

It also smooths out noisy gradients. Mini-batch training creates noise—momentum averages over recent updates, creating a smoother, more stable training trajectory.

**Where Momentum Struggles**

That same confidence can backfire. Momentum can overshoot the optimal point, especially near the end of training. It still uses the same learning rate for all parameters (though it adapts the effective step size through velocity). And you now have an extra hyperparameter (`β`) to tune.

**Best Used For:** Training deeper networks, escaping saddle points and local minima, when you want faster convergence than SGD but don't need per-parameter adaptation.

---

## Adam: The Adaptive Learner

**Algorithm:**
```python
m = β₁ * m + (1 - β₁) * grad
v = β₂ * v + (1 - β₂) * grad²
w = w - lr * (m / √v)
```

**Learner Analogy**

Picture a metacognitive learner—someone who doesn't just study, but tracks *how* they study. They notice: "I've spent 20 hours on calculus but only 2 on history. Maybe I should focus more on history now." They remember which topics they've practiced heavily and which are still unfamiliar, adjusting their effort accordingly.

They track both *direction* (what they're learning) and *intensity* (how much they've studied each topic). That's Adam.

**How It Works**

Adam (Adaptive Moment Estimation) combines two ideas:

1. **Momentum (`m`)**: Like Momentum SGD, it tracks the exponentially decaying average of past gradients. This is the "direction memory."

2. **Adaptive learning rates (`v`)**: It tracks the exponentially decaying average of past *squared* gradients. Parameters with large, consistent gradients get smaller effective learning rates. Parameters with small, infrequent gradients get larger effective learning rates.

The final update divides the momentum by the square root of the squared gradient accumulator, creating per-parameter adaptive learning rates.

**Variable Mapping:**
- `m` → Memory of which direction you've been learning (first moment)
- `v` → Memory of how intensely you've studied each topic (second moment)
- `β₁` → How much direction history to keep (typically 0.9)
- `β₂` → How much intensity history to keep (typically 0.999)
- `m / √v` → The balanced update that adapts per parameter

**What Adam Does Well**

Adam is the workhorse of modern deep learning. It works well out-of-the-box for most problems—language models, computer vision, reinforcement learning. By combining momentum with adaptive learning rates, it gets the benefits of both: fast convergence and per-parameter adaptation.

It's particularly robust to hyperparameter choices. The defaults (`β₁=0.9`, `β₂=0.999`, `lr=0.001`) work surprisingly well across diverse tasks.

**Where Adam Struggles**

Adam's aggressive adaptation can sometimes lead it to converge to "good enough" solutions rather than optimal ones. It finds local minima faster but sometimes those minima aren't as good as what SGD with momentum would eventually find with more time.

It also requires more memory—tracking both `m` and `v` for every parameter doubles the memory footprint compared to SGD. And while the defaults work well, you now have three hyperparameters to tune if you want to squeeze out extra performance.

**Best Used For:** Most deep learning tasks, when you want a robust default optimizer, training large models where per-parameter adaptation matters, when you don't have time for extensive hyperparameter tuning.

---

## The Big Picture

**Scale changes everything.** Full-batch gradient descent works for XOR's 4 examples but collapses at 60,000. Mini-batch SGD isn't a clever trick—it's survival at scale.

**Adaptive learning makes sense.** Not all weights should move equally. Some need bold steps, others careful nudges. Adam adjusts per-parameter instead of treating everything the same.

**No single optimizer is perfect.** SGD is reliable but stubborn. Momentum adds persistence but can overshoot. Adam balances everything but sometimes settles for "good enough."

Choose your optimizer like you'd choose a study strategy—match it to the problem you're solving.

---

    ## What's Next?

    There are many more optimizers in the wild: **AdaGrad** (adapts per-parameter but burns out on long training runs), **RMSProp** (fixes AdaGrad's aggressive decay), **AdaDelta** (removes the need for a learning rate), **NAdam** (Nesterov-accelerated Adam), **L-BFGS** (second-order method for smaller datasets), and newer variants like **AdamW** (Adam with weight decay done right), **RAdam** (rectified Adam with warmup), and **Lookahead** (maintains fast and slow weights).

    Each optimizer is a different learning strategy—a different way to navigate the loss landscape. I'll cover these in future posts when the context is right, showing you not just *what* they do, but *when* and *why* to use them.

    For now, you have the foundation: SGD grinds through problems with stubborn consistency, Momentum surges forward with accumulated confidence, and Adam adapts intelligently to each parameter's needs.

    Pick your learner. Start training.