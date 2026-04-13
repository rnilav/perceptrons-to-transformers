"""
RNN Playground — Post 8: Recurrent Networks & Sequential Data

One tab. One story.

A model reads a sentence word by word. At the end it must answer:
"Who was the subject?" Watch the model's confidence update live
as each word arrives — and see exactly where Vanilla RNN forgets
while LSTM holds on.

Run:
    streamlit run rnn_playground.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from rnn_models import (
    VanillaRNN, LSTM,
    SUBJ_NAMES, FILLER_WORDS, TRIGGER, N_SUBJ, VOCAB,
    make_sentence, train, eval_by_length, token_name,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RNN Playground", layout="wide")
st.title("🧠 Watch the Model Forget")
st.markdown(
    "A model reads a sentence one word at a time. "
    "At the end it must answer: **who was the subject?** "
    "Watch the model's confidence update at every step — "
    "and see exactly where Vanilla RNN forgets while LSTM holds on."
)

RNN_COLOR  = "#e07b39"
LSTM_COLOR = "#3a86ff"
CORRECT_COLOR = "#2ec4b6"
WRONG_COLOR   = "#e05252"


def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── sidebar: train / load ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown(
        "Models are trained on **short sentences** (2–5 filler words). "
        "The demo then tests them on **longer sentences** they've never seen — "
        "revealing where each model's memory breaks down."
    )
    hidden_size = st.select_slider("Hidden size", [16, 32, 48, 64], value=48)
    n_steps     = st.select_slider("Training steps", [2000, 4000, 6000], value=6000)
    train_btn   = st.button("🚀 Train both models", type="primary")

    if train_btn or "rnn_model" not in st.session_state:
        with st.spinner("Training Vanilla RNN…"):
            rnn = VanillaRNN(hidden_size=hidden_size, seed=7)
            rnn_hist = train(rnn, n_steps=n_steps)
            st.session_state["rnn_model"] = rnn
            st.session_state["rnn_hist"]  = rnn_hist

        with st.spinner("Training LSTM…"):
            lstm = LSTM(hidden_size=hidden_size, seed=7)
            lstm_hist = train(lstm, n_steps=n_steps)
            st.session_state["lstm_model"] = lstm
            st.session_state["lstm_hist"]  = lstm_hist

        st.success("✅ Both models trained.")

    if "rnn_model" in st.session_state:
        st.markdown("---")
        st.markdown("**Training accuracy (short sentences)**")
        rnn_final  = st.session_state["rnn_hist"][-1][1]
        lstm_final = st.session_state["lstm_hist"][-1][1]
        st.metric("Vanilla RNN", f"{rnn_final:.0%}")
        st.metric("LSTM",        f"{lstm_final:.0%}")


# ── main area ─────────────────────────────────────────────────────────────────
if "rnn_model" not in st.session_state:
    st.info("👈 Click **Train both models** in the sidebar to get started.")
    st.stop()

rnn  = st.session_state["rnn_model"]
lstm = st.session_state["lstm_model"]

# ── section 1: live sentence demo ─────────────────────────────────────────────
st.subheader("Part 1 — Watch the prediction update word by word")
st.markdown(
    "Pick a sentence length. The model reads it left to right. "
    "At each step, the bar chart shows its current best guess. "
    "**Short sentences**: both models are confident and correct. "
    "**Long sentences**: Vanilla RNN drifts. LSTM holds."
)

col_ctrl, col_demo = st.columns([1, 3])

with col_ctrl:
    n_filler = st.select_slider(
        "Sentence length (filler words)",
        options=[2, 4, 6, 8, 10, 12, 15],
        value=12,
    )
    demo_seed = st.number_input("Sentence seed", 0, 999, 42)
    run_demo  = st.button("▶  Show sentence", type="primary")

with col_demo:
    if run_demo:
        rng_demo = np.random.default_rng(int(demo_seed))
        xs, subject, tokens = make_sentence(n_filler, rng_demo)
        n_tokens = len(tokens)

        preds_rnn  = rnn.predict_steps(xs)
        preds_lstm = lstm.predict_steps(xs)

        # ── build the step-by-step figure ────────────────────────────────
        fig, axes = plt.subplots(
            n_tokens, 2,
            figsize=(11, max(4, n_tokens * 0.75)),
            gridspec_kw={"wspace": 0.35},
        )
        if n_tokens == 1:
            axes = [axes]

        for step_i, (tok_idx, pr, pl) in enumerate(
            zip(tokens, preds_rnn, preds_lstm)
        ):
            tok_label = token_name(tok_idx)
            rnn_guess  = np.argmax(pr)
            lstm_guess = np.argmax(pl)

            for col_i, (probs, guess, color, model_label) in enumerate([
                (pr, rnn_guess,  RNN_COLOR,  "Vanilla RNN"),
                (pl, lstm_guess, LSTM_COLOR, "LSTM"),
            ]):
                ax = axes[step_i][col_i]
                bar_colors = [
                    CORRECT_COLOR if j == subject else
                    WRONG_COLOR   if j == guess and j != subject else
                    "#dddddd"
                    for j in range(N_SUBJ)
                ]
                bars = ax.barh(
                    range(N_SUBJ), probs,
                    color=bar_colors, height=0.6, edgecolor="none",
                )
                ax.set_xlim(0, 1.05)
                ax.set_yticks(range(N_SUBJ))
                ax.set_yticklabels(SUBJ_NAMES, fontsize=7)
                ax.tick_params(axis="x", labelsize=6)
                ax.axvline(0.5, color="#cccccc", linewidth=0.5, linestyle="--")

                # row label on left model only
                if col_i == 0:
                    is_trigger = tok_idx == TRIGGER
                    row_label  = f"{'❓' if is_trigger else '→'} {tok_label}"
                    ax.set_ylabel(row_label, fontsize=8, rotation=0,
                                  labelpad=55, va="center")

                # column header on first row
                if step_i == 0:
                    ax.set_title(model_label, fontsize=9,
                                 color=color, fontweight="bold")

                # mark correct answer
                ax.text(
                    probs[subject] + 0.02, subject,
                    f"{probs[subject]:.0%}",
                    va="center", fontsize=6.5,
                    color=CORRECT_COLOR if guess == subject else "#888888",
                )

        fig.suptitle(
            f"Subject: {SUBJ_NAMES[subject]}  |  "
            f"{n_filler} filler words  |  "
            f"trained on 2–5 fillers",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()
        st.image(fig_to_image(fig), width='stretch')

        # ── confidence of correct answer over time ────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        steps_x = list(range(1, n_tokens + 1))
        conf_rnn  = [pr[subject] for pr in preds_rnn]
        conf_lstm = [pl[subject] for pl in preds_lstm]

        ax2.plot(steps_x, conf_rnn,  color=RNN_COLOR,  linewidth=2.5,
                 marker="o", markersize=5, label="Vanilla RNN")
        ax2.plot(steps_x, conf_lstm, color=LSTM_COLOR, linewidth=2.5,
                 marker="o", markersize=5, label="LSTM")
        ax2.axhline(1/N_SUBJ, color="gray", linewidth=1,
                    linestyle=":", label=f"chance ({1/N_SUBJ:.0%})")
        ax2.axvline(1, color="#aaaaaa", linewidth=1,
                    linestyle="--", alpha=0.6, label="subject seen (step 1)")

        ax2.set_xticks(steps_x)
        ax2.set_xticklabels(
            [token_name(t) for t in tokens], rotation=35, ha="right", fontsize=8
        )
        ax2.set_ylabel(f"P({SUBJ_NAMES[subject]})", fontsize=10)
        ax2.set_ylim(0, 1.1)
        ax2.set_title(
            f"Confidence in correct answer ({SUBJ_NAMES[subject]}) over time",
            fontsize=11,
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        st.image(fig_to_image(fig2), width='stretch')

        # ── verdict ───────────────────────────────────────────────────────
        rnn_final_guess  = np.argmax(preds_rnn[-1])
        lstm_final_guess = np.argmax(preds_lstm[-1])
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if rnn_final_guess == subject:
                st.success(f"Vanilla RNN: ✅ {SUBJ_NAMES[rnn_final_guess]}")
            else:
                st.error(
                    f"Vanilla RNN: ❌ guessed {SUBJ_NAMES[rnn_final_guess]} "
                    f"(correct: {SUBJ_NAMES[subject]})"
                )
        with col_v2:
            if lstm_final_guess == subject:
                st.success(f"LSTM: ✅ {SUBJ_NAMES[lstm_final_guess]}")
            else:
                st.error(
                    f"LSTM: ❌ guessed {SUBJ_NAMES[lstm_final_guess]} "
                    f"(correct: {SUBJ_NAMES[subject]})"
                )
    else:
        st.info("Choose a sentence length and click **Show sentence**.")


# ── section 2: accuracy vs sentence length ────────────────────────────────────
st.markdown("---")
st.subheader("Part 2 — Where does each model's memory break down?")
st.markdown(
    "Both models were trained only on **short sentences** (2–5 filler words). "
    "Here we test them on sentences of increasing length — "
    "including lengths they've never seen during training."
)

run_curve = st.button("▶  Run accuracy vs sentence length", type="primary")

if run_curve:
    test_lengths = [2, 4, 6, 8, 10, 12, 15, 18, 22]
    with st.spinner("Evaluating…"):
        accs_rnn  = eval_by_length(rnn,  test_lengths)
        accs_lstm = eval_by_length(lstm, test_lengths)

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(test_lengths, accs_rnn,  color=RNN_COLOR,  linewidth=2.5,
             marker="o", markersize=7, label="Vanilla RNN")
    ax3.plot(test_lengths, accs_lstm, color=LSTM_COLOR, linewidth=2.5,
             marker="o", markersize=7, label="LSTM")
    ax3.axhline(1/N_SUBJ, color="gray", linewidth=1,
                linestyle=":", label=f"chance ({1/N_SUBJ:.0%})")
    ax3.axvspan(0, 5.5, alpha=0.06, color="green", label="trained range")

    ax3.set_xlabel("Number of filler words", fontsize=12)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title(
        "Accuracy vs Sentence Length — trained on 2–5 fillers, tested beyond",
        fontsize=12, fontweight="bold",
    )
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks(test_lengths)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    st.image(fig_to_image(fig3), width='stretch')

    st.caption(
        "Inside the green band: both models were trained here — both do well. "
        "Beyond it: Vanilla RNN's accuracy drops toward chance as the sentence grows. "
        "LSTM holds — its cell state carries the subject identity across many steps "
        "without it decaying."
    )
else:
    st.info("Click **Run accuracy vs sentence length** to see the full picture.")
