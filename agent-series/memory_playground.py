"""
THE AGENT'S MIND — Interactive Memory Playground
=================================================
Watch memory form in real-time as you chat with an agent.
No LLM — pure scripted mechanics showing real memory operations.

Run: streamlit run memory_playground.py
"""

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Agent's Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────────────────────

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.memory_on = True
    st.session_state.session_num = 1
    st.session_state.working_memory = []
    st.session_state.episodic = []
    st.session_state.semantic = []
    st.session_state.procedural = []
    st.session_state.last_ops = []
    st.session_state.last_retrievals = []

# ──────────────────────────────────────────────────────────────────────────────
# Scripted Conversation
# ──────────────────────────────────────────────────────────────────────────────

CONVERSATION = [
    {
        "user": "Hi! I'm Nilav. I'm planning a trip to Japan in March.",
        "agent": "Hello Nilav! Japan in March — that's cherry blossom season. Wonderful choice. What would you like help with?",
        "ops": [
            {"type": "semantic", "display": "(Nilav) →[traveling_to]→ (Japan)"},
            {"type": "semantic", "display": "(trip) →[month]→ (March)"},
            {"type": "episodic", "display": "Session started: Nilav planning Japan trip for March", "importance": 0.7},
        ]
    },
    {
        "user": "I'm vegetarian and allergic to shellfish. Important to know!",
        "agent": "Noted — vegetarian diet, shellfish allergy. I'll make sure every recommendation respects both. Safety first.",
        "ops": [
            {"type": "semantic", "display": "(Nilav) →[diet]→ (vegetarian)"},
            {"type": "semantic", "display": "(Nilav) →[allergy]→ (shellfish)"},
            {"type": "episodic", "display": "Critical: vegetarian + shellfish allergy", "importance": 0.95},
        ]
    },
    {
        "user": "Budget is around $800 for flights from Singapore.",
        "agent": "Got it — Singapore to Japan, under $800. Direct flights on Scoot or ANA should work for March!",
        "ops": [
            {"type": "semantic", "display": "(trip) →[budget]→ ($800)"},
            {"type": "semantic", "display": "(Nilav) →[located_in]→ (Singapore)"},
        ]
    },
    {
        "user": "Can you recommend restaurants in Tokyo for me?",
        "agent": "For Tokyo restaurants, I'm filtering: ✓ vegetarian, ✗ no shellfish, 📍 central Tokyo. Here are my top picks...",
        "ops": [
            {"type": "procedural", "display": "WHEN: restaurant request → THEN: 1) check diet 2) check allergies 3) filter location"},
            {"type": "episodic", "display": "Recommended Tokyo restaurants (filtered)", "importance": 0.6},
        ]
    },
    {
        "user": "Wait — what did I tell you about my food restrictions?",
        "agent": "You mentioned two things: you're vegetarian, and you have a shellfish allergy. I've been factoring both into every recommendation.",
        "ops": [
            {"type": "retrieval", "display": "Retrieved: (Nilav)→[diet]→(vegetarian) + (Nilav)→[allergy]→(shellfish)", "source": "semantic"},
        ]
    },
    {
        "user": "Perfect. Also suggest vegetarian ramen spots specifically?",
        "agent": "Applying my restaurant procedure again — vegetarian ramen, no shellfish broth, Tokyo area. Here are specialist spots...",
        "ops": [
            {"type": "procedural_reuse", "display": "Reused restaurant procedure (2nd time — faster!)"},
            {"type": "episodic", "display": "Nilav interested in vegetarian ramen specifically", "importance": 0.5},
        ]
    },
]

SESSION_2 = [
    {
        "user": "Hey, I'm back. Remember me?",
        "agent_on": "Welcome back, Nilav! Of course — we were planning your Japan trip for March. Vegetarian, shellfish allergy, $800 flight budget from Singapore. How can I help today?",
        "agent_off": "Hello! I'm your travel assistant. How can I help you today?",
        "ops_on": [{"type": "retrieval", "display": "Retrieved: Nilav, Japan, March, vegetarian, shellfish, $800, Singapore", "source": "all long-term stores"}],
        "ops_off": []
    },
    {
        "user": "What restaurants did we talk about last time?",
        "agent_on": "Last session we discussed vegetarian restaurants in Tokyo — specifically ramen spots without shellfish broth. Want me to continue from there?",
        "agent_off": "I'm sorry, I don't have any record of previous conversations. Could you tell me what you're looking for?",
        "ops_on": [{"type": "retrieval", "display": "Retrieved: restaurant discussion + ramen interest from episodic memory", "source": "episodic"}],
        "ops_off": []
    },
]


def process_step():
    """Process current step — add to working memory and run memory ops."""
    if st.session_state.session_num == 1:
        conv = CONVERSATION
    else:
        conv = SESSION_2

    if st.session_state.step >= len(conv):
        return

    step_data = conv[st.session_state.step]

    # Add to working memory
    st.session_state.working_memory.append({"role": "user", "text": step_data["user"]})

    if st.session_state.session_num == 1:
        agent_text = step_data["agent"]
        ops = step_data["ops"]
    else:
        agent_text = step_data["agent_on"] if st.session_state.memory_on else step_data["agent_off"]
        ops = step_data.get("ops_on", []) if st.session_state.memory_on else step_data.get("ops_off", [])

    st.session_state.working_memory.append({"role": "agent", "text": agent_text})

    # Execute memory operations
    st.session_state.last_ops = []
    st.session_state.last_retrievals = []

    if st.session_state.memory_on:
        for op in ops:
            if op["type"] == "semantic":
                st.session_state.semantic.append(op["display"])
                st.session_state.last_ops.append(("semantic", op["display"]))
            elif op["type"] == "episodic":
                st.session_state.episodic.append({"text": op["display"], "importance": op.get("importance", 0.5)})
                st.session_state.last_ops.append(("episodic", op["display"]))
            elif op["type"] == "procedural":
                st.session_state.procedural.append({"text": op["display"], "uses": 1})
                st.session_state.last_ops.append(("procedural", op["display"]))
            elif op["type"] == "procedural_reuse":
                if st.session_state.procedural:
                    st.session_state.procedural[-1]["uses"] += 1
                st.session_state.last_ops.append(("procedural", op["display"]))
            elif op["type"] == "retrieval":
                st.session_state.last_retrievals.append(op["display"])

    # Trim working memory (simulate context window overflow)
    max_turns = 8
    if len(st.session_state.working_memory) > max_turns:
        st.session_state.working_memory = st.session_state.working_memory[-max_turns:]

    st.session_state.step += 1


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Controls & Guide
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Controls")
    st.markdown("---")

    st.session_state.memory_on = st.toggle("Long-term memory ON", value=st.session_state.memory_on)

    st.markdown("---")

    if st.button("📨 Send Next Message", use_container_width=True, type="primary"):
        process_step()
        st.rerun()

    conv = CONVERSATION if st.session_state.session_num == 1 else SESSION_2
    remaining = len(conv) - st.session_state.step
    st.caption(f"Session {st.session_state.session_num} · {remaining} messages left")

    if st.session_state.step > 0 and st.session_state.step < len(conv):
        next_msg = conv[st.session_state.step]["user"]
        st.info(f"**Next:** \"{next_msg[:80]}\"")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.working_memory = []
            st.session_state.step = 0
            st.session_state.session_num = 2
            st.session_state.last_ops = []
            st.session_state.last_retrievals = []
            st.rerun()
    with col2:
        if st.button("🗑️ Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 How to use")
    st.markdown("""
1. Click **Send Next Message** to advance the conversation  
2. Watch the **right panel** — memory forms in real-time  
3. Complete Session 1, then click **New Session**  
4. With memory ON → agent remembers you  
5. Toggle memory OFF → agent becomes a stranger  
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🧠 The Agent's Mind")
st.caption("Watch how information flows from short-term working memory into long-term storage — in real time.")

# Two main columns
col_chat, col_mind = st.columns([2, 3], gap="large")

# ── LEFT: Conversation ──
with col_chat:
    st.markdown("### 💬 Conversation")

    if not st.session_state.working_memory:
        st.info("👆 Click **Send Next Message** in the sidebar to start chatting.")
    else:
        for msg in st.session_state.working_memory:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.write(msg["text"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["text"])

    # Completion messages
    conv = CONVERSATION if st.session_state.session_num == 1 else SESSION_2
    if st.session_state.step >= len(conv) and st.session_state.working_memory:
        if st.session_state.session_num == 1:
            st.success("✅ Session 1 complete! Click **New Session** in the sidebar to see what persists across sessions.")
        else:
            if st.session_state.memory_on:
                st.success("✅ The agent remembered everything from last session — because long-term memory persisted.")
            else:
                st.error("❌ The agent forgot you completely — no long-term memory means every session starts from zero.")


# ── RIGHT: The Agent's Mind ──
with col_mind:
    st.markdown("### 🧠 Agent's Memory State")

    # ─── Working Memory ───
    st.markdown("#### 📝 Working Memory (Short-term)")
    capacity = len(st.session_state.working_memory)
    max_cap = 8
    pct = capacity / max_cap

    if pct < 0.5:
        bar_color = "green"
    elif pct < 0.8:
        bar_color = "orange"
    else:
        bar_color = "red"

    st.progress(pct, text=f"{capacity}/{max_cap} turns used")

    if not st.session_state.working_memory:
        st.caption("_Empty — send a message to start_")
    else:
        with st.expander(f"View context window ({capacity} turns)", expanded=False):
            for msg in st.session_state.working_memory:
                icon = "🧑" if msg["role"] == "user" else "🤖"
                st.text(f"{icon} {msg['text'][:90]}")

    # ─── Flow Indicator ───
    st.markdown("---")
    if st.session_state.last_ops:
        st.markdown("#### ⬇️ Consolidating to Long-term")
        for (op_type, display) in st.session_state.last_ops:
            color_map = {"semantic": "🟢", "episodic": "🟠", "procedural": "🟡"}
            label_map = {"semantic": "→ Semantic", "episodic": "→ Episodic", "procedural": "→ Procedural"}
            icon = color_map.get(op_type, "⚪")
            label = label_map.get(op_type, "→ Stored")
            st.markdown(f"{icon} **{label}:** {display}")
    elif st.session_state.last_retrievals:
        st.markdown("#### ⬆️ Retrieving from Long-term")
        for r in st.session_state.last_retrievals:
            st.markdown(f"🔵 **Retrieved:** {r}")
    else:
        st.markdown("#### ↕️ Flow")
        st.caption("_Send a message to see memory operations_")

    st.markdown("---")

    # ─── Long-term Memory ───
    st.markdown("#### 🗄️ Long-term Memory")

    tab_epi, tab_sem, tab_proc = st.tabs(["📚 Episodic", "🕸️ Semantic", "🎹 Procedural"])

    with tab_epi:
        if not st.session_state.episodic:
            st.caption("_No events stored yet_")
        else:
            for item in reversed(st.session_state.episodic):
                imp = item["importance"]
                if imp > 0.8:
                    imp_label = "🔴 HIGH"
                elif imp > 0.5:
                    imp_label = "🟡 MED"
                else:
                    imp_label = "⚪ LOW"
                st.markdown(f"**{imp_label}** · {item['text']}")

    with tab_sem:
        if not st.session_state.semantic:
            st.caption("_No facts extracted yet_")
        else:
            for fact in st.session_state.semantic:
                st.code(fact, language=None)

    with tab_proc:
        if not st.session_state.procedural:
            st.caption("_No procedures learned yet_")
        else:
            for proc in st.session_state.procedural:
                st.markdown(f"```\n{proc['text']}\n```")
                st.caption(f"Used {proc['uses']}×")
