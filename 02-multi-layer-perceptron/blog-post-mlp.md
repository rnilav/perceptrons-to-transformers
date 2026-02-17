_"The perceptron has many limitations... the most serious is its inability to learn even the simplest nonlinear functions."_ **Marvin Minsky**

## **The Problem That Stumped AI**

In my last post, I mentioned that the perceptron could learn AND, OR, and NAND gates perfectly. But there was one simple logic gate it couldn't learn, no matter how much you trained it.

That gate was XOR (exclusive-or).

```
XOR Truth Table:
┌─────────┬─────────┬────────┐
│ Input 1 │ Input 2 │ Output │
├─────────┼─────────┼────────┤
│    0    │    0    │   0    │
│    0    │    1    │   1    │
│    1    │    0    │   1    │
│    1    │    1    │   0    │
└─────────┴─────────┴────────┘
```

When Marvin Minsky and Seymour Papert published their book "Perceptrons" in 1969, they proved mathematically that single-layer perceptrons couldn't solve XOR. This revelation triggered the first "AI winter" - funding dried up, research stalled, and neural networks were largely abandoned for over a decade.

But why? What makes XOR so special?

## **The Geometry of Impossibility**

Here's the thing: a perceptron draws a straight line to separate classes. That's it. One straight line.

For XOR, you need the output to be 1 when inputs are different, and 0 when they're the same:

```
Visual representation:
    Input 2
      ↑
    1 │  [1]    [0]
      │
    0 │  [0]    [1]
      └──────────────→ Input 1
         0       1

[0] = Output 0 
[1] = Output 1
```

Try drawing a single straight line that separates the red squares from the blue circles. You can't. The pattern is diagonal - you'd need two lines, or a curve.

This is what "not linearly separable" means.

I spent hours staring at this diagram when I first learned about it. I tried every angle, every position for that line. Nothing worked. And that's exactly the point - it's mathematically impossible. The perceptron's limitation isn't a bug, it's a fundamental constraint of linear classifiers.

For AND and OR gates, the pattern is simple - all the 1s are on one side, all the 0s on the other. But XOR? The classes are interleaved. You need a more sophisticated approach.

## **The Breakthrough: Hidden Layers**

When I was a kid learning math, adding single-digit numbers was simple. 3 + 5 = 8. I just did it. One step, done.

But then came the leap to multi-digit addition: 27 + 15.

I kept getting it wrong. I'd add 2 + 1 = 3, then 7 + 5 = 12, and write 312. Completely wrong. My brain was treating it like two separate single-digit problems mashed together. I was missing something invisible.

Then came the breakthrough: 7 + 5 doesn't just equal 12. It creates a 1 that carries over to the next column. That invisible 1 moving from ones to tens column—that was the missing piece. Once I understood the carry, it clicked.

The carry was an intermediate step that transformed the problem.
It sounds trivial now. But back then? It was a massive leap for me. I couldn't see why single-digit rules didn't just scale up. I needed something new—not more of the same.

**That's the hidden layer. But here's the catch:**

If I just wrote down two addition problems and stacked them on top of each other, nothing changes. 2 + 1, then 7 + 5. That's still just two separate additions. Adding more steps doesn't help if each step is the same linear operation.

But the carry isn't linear. When 7 + 5 = 12, something special happens: the 1 doesn't stay in that column. It transforms—it becomes a 1 in a different column, changing what comes next. That transformation—that non-linearity—is what makes the whole system work.

Without the carry's transformation, stacking problems is useless. With it, multi-digit addition becomes possible.

That's exactly what non-linear activation functions do in neural networks.

A single-layer perceptron is like single-digit addition—inputs straight to output, no transformation. If you just stack more linear layers, you still have the same problem: one straight line, no matter how many you combine.

But add non-linear activation functions (like sigmoid or ReLU)—the carry transforms the space. Now XOR becomes solvable.

![2-2-1 Multi-Layer Network](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/y9148fmbr0dvvu9qp8kb.png)

**Note:** Weights/Biases marked in the above diagram are hand-crafted specifically for the XOR problem

## **Solving XOR: The Aha Moment**

With a 2-2-1 network (2 inputs, 2 hidden neurons, 1 output), we can finally solve XOR:

```
How it works:
┌──────────────────────────────────────┐
│ Hidden Neuron 1: Learns OR pattern   │
│   (fires when x₁ OR x₂ is 1)         │
│                                      │
│ Hidden Neuron 2: Learns AND pattern  │
│   (fires when x₁ AND x₂ are 1)       │
│                                      │
│ Output: Combines them                │
│   (OR but NOT AND = XOR)             │
└──────────────────────────────────────┘
```

When I first ran this code and saw it work, I got it. The hidden layer isn't just adding complexity - it's transforming the problem into something solvable.

_A comparative snapshot generated in the playground_

![single vs multi layer perceptron](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/4wlf9ds9vfyxdfmr8y4d.png)

**Try it yourself:** Run the interactive playground to see the curved decision boundary in action. Adjust the weight slider to see how the boundary changes from weak to strong. Compare it with a perceptron's straight line attempt. The visualisation makes it clear why hidden layers are the breakthrough.

**GitHub Repository:** [perceptrons-to-transformers - 02-xor-problem](https://github.com/rnilav/perceptrons-to-transformers/tree/main/02-xor-problem)

What you'll find:
- `02-multi-layer-perceptron/mlp.py` - Clean MLP implementation
- `02-multi-layer-perceptron/mlp_playground.py` - Interactive Streamlit app

The playground lets you:
- See the curved decision boundary that solves XOR
- Adjust weights and watch the boundary change in real-time
- View the network architecture with all weights labeled
- Compare perceptron's straight line vs MLP's curve

## **What This Unlocked**

Solving XOR might seem trivial now. But it was the breakthrough that unlocked everything.

The problem wasn't just XOR. It was the realisation: hidden layers don't just add complexity—they enable non-linear thinking. Once researchers understood this, the floodgates opened.

In the 1980s, Geoffrey Hinton, David Rumelhart, and Ronald Williams proved you could actually train these multi-layer networks with backpropagation. Suddenly, problems that seemed impossible became solvable. The AI winter thawed.

The progression is beautiful:

**Perceptrons** learned to draw lines (linear boundaries)
**MLPs** learned to draw curves (non-linear boundaries)
**Deep networks** learned hierarchies (edges → shapes → objects → concepts)

Today's neural networks—from image classifiers to GPT-4—all follow the same principle: stack layers with non-linear activations to transform data into increasingly meaningful representations.
It all started with one insight: add a hidden layer.

All from adding that first hidden layer.

## **What's Next: **

We can now build networks that solve XOR. But there's one crucial question: How do we learn the weights?

The XOR network I showed you uses hand-crafted weights—I manually set values that worked. But for real problems with thousands of inputs and millions of weights, we can't do that by hand.
We need an algorithm that automatically learns the right weights from examples.

That algorithm is called backpropagation, and it's what makes neural networks practical. It's how networks learn from their mistakes and gradually improve.

In the next post, we'll dive into backpropagation—the algorithm that ties everything together. It involves calculus, but I promise to make it intuitive.

---

## References

1. **Minsky, M., & Papert, S.** (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

2. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*. Determination Press. Available at: http://neuralnetworksanddeeplearning.com/

---

**Tags:** #MachineLearning #AI #DeepLearning #NeuralNetworks #MLP

**Series:** From Perceptrons to Transformers - Part 2 of 18

**Code:** [GitHub Repository](https://github.com/rnilav/perceptrons-to-transformers)
