"""
Vanilla RNN and LSTM from scratch — used by the playground demo.

Task: given a sequence [subject, filler, filler, ..., TRIGGER],
predict which subject appeared at the start.

Vocab layout:
  0-5   → 6 subjects (Alice, Bob, Carol, David, Emma, Frank)
  6-11  → 6 filler tokens (words)
  12    → TRIGGER ("who?")

Post 8: Recurrent Networks & Sequential Data
Series: From Perceptrons to Transformers
"""

import numpy as np

VOCAB     = 13
N_SUBJ    = 6
TRIGGER   = 12
SUBJ_NAMES  = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
FILLER_WORDS = ["walked", "to", "the", "store", "and", "bought"]


# ── activations ──────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── data generation ───────────────────────────────────────────────────────────

def make_sentence(n_filler, rng):
    """
    [subject, filler×n, TRIGGER]
    Returns (xs, subject_idx, token_indices).
    """
    subject = int(rng.integers(0, N_SUBJ))
    fillers = [int(rng.integers(N_SUBJ, VOCAB - 1)) for _ in range(n_filler)]
    tokens  = [subject] + fillers + [TRIGGER]
    xs      = [np.eye(VOCAB)[t] for t in tokens]
    return xs, subject, tokens


def token_name(idx):
    if idx < N_SUBJ:
        return SUBJ_NAMES[idx]
    if idx == TRIGGER:
        return "WHO?"
    return FILLER_WORDS[idx - N_SUBJ]


# ── Vanilla RNN ───────────────────────────────────────────────────────────────

class VanillaRNN:
    name = "Vanilla RNN"

    def __init__(self, hidden_size=48, seed=7):
        rng = np.random.default_rng(seed)
        H, V, O = hidden_size, VOCAB, N_SUBJ
        s = 0.1
        self.W_x = rng.normal(0, s, (H, V))
        self.W_h = rng.normal(0, s, (H, H))
        self.b_h = np.zeros(H)
        self.W_o = rng.normal(0, s, (O, H))
        self.b_o = np.zeros(O)
        self.n   = H

    def predict_steps(self, xs):
        """Return softmax prediction after each token."""
        h = np.zeros(self.n)
        preds = []
        for x in xs:
            h = np.tanh(self.W_x @ x + self.W_h @ h + self.b_h)
            logits = self.W_o @ h + self.b_o
            preds.append(softmax(logits).copy())
        return preds

    def _forward_cache(self, xs):
        h = np.zeros(self.n)
        hs, pre = [h.copy()], []
        for x in xs:
            z = self.W_x @ x + self.W_h @ h + self.b_h
            h = np.tanh(z)
            hs.append(h.copy())
            pre.append(z)
        return self.W_o @ h + self.b_o, hs, pre

    def train_step(self, xs, target, lr=0.02, clip=5.0):
        logits, hs, pre = self._forward_cache(xs)
        probs = softmax(logits); probs[target] -= 1.0
        dW_o = np.outer(probs, hs[-1]); db_o = probs.copy()
        dh   = self.W_o.T @ probs
        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        db_h = np.zeros_like(self.b_h)
        for t in reversed(range(len(xs))):
            dt = (1 - np.tanh(pre[t]) ** 2) * dh
            dW_x += np.outer(dt, xs[t])
            dW_h += np.outer(dt, hs[t])
            db_h += dt
            dh    = self.W_h.T @ dt
        for g in [dW_x, dW_h, db_h, dW_o, db_o]:
            np.clip(g, -clip, clip, out=g)
        self.W_x -= lr * dW_x; self.W_h -= lr * dW_h; self.b_h -= lr * db_h
        self.W_o -= lr * dW_o; self.b_o -= lr * db_o
        return -np.log(softmax(logits)[target] + 1e-9)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTM:
    name = "LSTM"

    def __init__(self, hidden_size=48, seed=7):
        rng = np.random.default_rng(seed)
        n, m = hidden_size, VOCAB
        self.W  = rng.normal(0, 0.1, (4 * n, n + m))
        self.b  = np.zeros(4 * n)
        self.b[:n] = 1.0          # forget gate bias → remember by default
        self.W_o = rng.normal(0, 0.1, (N_SUBJ, n))
        self.b_o = np.zeros(N_SUBJ)
        self.n   = n

    def predict_steps(self, xs):
        n = self.n
        h, c = np.zeros(n), np.zeros(n)
        preds = []
        for x in xs:
            comb = np.concatenate([h, x])
            raw  = self.W @ comb + self.b
            f = sigmoid(raw[0*n:1*n]); i = sigmoid(raw[1*n:2*n])
            g = np.tanh(raw[2*n:3*n]); o = sigmoid(raw[3*n:4*n])
            c = f * c + i * g
            h = o * np.tanh(c)
            logits = self.W_o @ h + self.b_o
            preds.append(softmax(logits).copy())
        return preds

    def train_step(self, xs, target, lr=0.02, clip=5.0):
        n = self.n
        h, c = np.zeros(n), np.zeros(n)
        cache, hs, cs = [], [h.copy()], [c.copy()]
        for x in xs:
            comb = np.concatenate([h, x])
            raw  = self.W @ comb + self.b
            f = sigmoid(raw[0*n:1*n]); i = sigmoid(raw[1*n:2*n])
            g = np.tanh(raw[2*n:3*n]); o = sigmoid(raw[3*n:4*n])
            c = f * c + i * g; tc = np.tanh(c); h = o * tc
            hs.append(h.copy()); cs.append(c.copy())
            cache.append((comb, f, i, g, o, tc, cs[-2]))
        logits = self.W_o @ h + self.b_o
        probs  = softmax(logits); probs[target] -= 1.0
        dW_o   = np.outer(probs, hs[-1]); db_o = probs.copy()
        dh     = self.W_o.T @ probs; dc = np.zeros(n)
        dW     = np.zeros_like(self.W); db = np.zeros_like(self.b)
        for t in reversed(range(len(xs))):
            comb, f, ig, g, o, tc, cp = cache[t]
            dc2  = dc + dh * o * (1 - tc ** 2)
            draw = np.concatenate([
                dc2 * cp * f * (1 - f),
                dc2 * g  * ig * (1 - ig),
                dc2 * ig * (1 - g ** 2),
                dh  * tc * o  * (1 - o),
            ])
            dW  += np.outer(draw, comb); db += draw
            dc_  = self.W.T @ draw; dh = dc_[:n]; dc = dc2 * f
        for g_ in [dW, db, dW_o, db_o]:
            np.clip(g_, -clip, clip, out=g_)
        self.W  -= lr * dW;  self.b  -= lr * db
        self.W_o -= lr * dW_o; self.b_o -= lr * db_o
        return -np.log(softmax(logits)[target] + 1e-9)


# ── training ──────────────────────────────────────────────────────────────────

def train(model, n_steps=6000, train_lengths=(2, 3, 4, 5), lr=0.02, seed=7):
    """
    Train model on the subject-recall task.
    Returns list of (step, accuracy) checkpoints.
    """
    rng_tr = np.random.default_rng(seed)
    rng_ev = np.random.default_rng(seed + 999)
    history = []
    for step in range(n_steps):
        n_fill = train_lengths[step % len(train_lengths)]
        xs, t, _ = make_sentence(n_fill, rng_tr)
        model.train_step(xs, t, lr=lr)
        if step % 300 == 0:
            correct = 0
            for length in train_lengths:
                for _ in range(40):
                    xs_e, t_e, _ = make_sentence(length, rng_ev)
                    preds = model.predict_steps(xs_e)
                    if np.argmax(preds[-1]) == t_e:
                        correct += 1
            history.append((step, correct / (len(train_lengths) * 40)))
    return history


def eval_by_length(model, lengths, n_per_length=300, seed=500):
    """Return accuracy at each sentence length."""
    accs = []
    for n_fill in lengths:
        rng = np.random.default_rng(seed + n_fill)
        correct = 0
        for _ in range(n_per_length):
            xs, t, _ = make_sentence(n_fill, rng)
            if np.argmax(model.predict_steps(xs)[-1]) == t:
                correct += 1
        accs.append(correct / n_per_length)
    return accs
