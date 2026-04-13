# Post 8: Recurrent Networks & Sequential Data

Interactive playground for the RNN blog post. One demo, one story: watch a model read a sentence word by word and see exactly where it forgets.

## Setup

```bash
pip install -r requirements.txt
streamlit run rnn_playground.py
```

---

## The Demo

Both models (Vanilla RNN and LSTM) are trained on the same task: read a sequence of words, then answer "who was the subject?" They're trained only on **short sentences** (2–5 filler words). The demo then tests them on longer sentences they've never seen.

### Part 1 — Watch the prediction update word by word

Pick a sentence length and hit **Show sentence**. You get:

- A step-by-step bar chart showing each model's confidence in every possible subject after each word arrives
- A confidence line chart tracking how sure each model is about the correct answer over time

**What to observe:**

- At short lengths (2–4 fillers): both models stay confident and correct — the subject is recent enough that even vanilla RNN remembers it
- At medium lengths (8–10 fillers): vanilla RNN's confidence starts to wobble — it may drift to a wrong guess mid-sentence before recovering
- At long lengths (12–15 fillers): vanilla RNN often switches its guess entirely before the final "WHO?" token. LSTM stays locked on the correct subject with high confidence throughout
- The confidence line chart makes the forgetting visible — vanilla RNN's line drops and oscillates; LSTM's line rises after step 1 and stays high

Try different sentence seeds to see different examples. Some seeds will show vanilla RNN forgetting dramatically; others it holds. LSTM is consistently stable.

### Part 2 — Accuracy vs sentence length

Hit **Run accuracy vs sentence length** to see the full picture across 300 sentences at each length.

**What to observe:**

- Inside the green band (trained range): both models score near 100%
- Beyond it: vanilla RNN drops toward the chance line (17%) as sentences get longer
- LSTM holds well past the trained range — its cell state carries the subject identity without it decaying
- The crossover point (where vanilla RNN drops below ~70%) is typically around 10–12 filler words

---

## Files

| File | Purpose |
|---|---|
| `rnn_models.py` | VanillaRNN and LSTM with BPTT, training and eval utilities |
| `rnn_playground.py` | Streamlit app |
| `blog-post-rnn.md` | The blog post this accompanies |
| `requirements.txt` | Dependencies |
