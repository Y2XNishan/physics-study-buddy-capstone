import json
import re
from uuid import uuid4

import streamlit as st

def format_physics_formulas(text: str) -> str:
    """Wrap common physics formulas in LaTeX formatting for Streamlit display."""
    formulas = [
        (r"\bV\s*=\s*IR\b", r"$V = IR$"),
        (r"\bF\s*=\s*ma\b", r"$F = ma$"),
        (r"\bv\s*=\s*u\s*\+\s*at\b", r"$v = u + at$"),
        (r"\bs\s*=\s*ut\s*\+\s*1/2\s*at\^2\b", r"$s = ut + \\frac{1}{2}at^2$"),
        (r"\bv\^2\s*=\s*u\^2\s*\+\s*2as\b", r"$v^2 = u^2 + 2as$"),
        (r"\bE\s*=\s*mc\^2\b", r"$E = mc^2$"),
        (r"\b1/2\s*mv\^2\b", r"$\\frac{1}{2}mv^2$"),
        (r"\bmgh\b", r"$mgh$"),
        (r"\bv\s*=\s*f\s*lambda\b", r"$v = f \lambda$"),
        (r"\bn1\s*sin\(theta1\)\s*=\s*n2\s*sin\(theta2\)", r"$n_1 \sin\theta_1 = n_2 \sin\theta_2$"),
    ]
    formatted = text
    for pattern, replacement in formulas:
        formatted = re.sub(pattern, replacement, formatted)
    return formatted

from physics_study_buddy.agent_core import build_agent


@st.cache_resource
def load_agent():
    return build_agent()


def reset_conversation() -> None:
    if "thread_id" in st.session_state and "agent" in globals():
        load_agent().reset_thread(st.session_state.thread_id)
    st.session_state.thread_id = f"physics-ui-{uuid4()}"
    st.session_state.messages = []


st.set_page_config(page_title="Physics Study Buddy", page_icon="⚛️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .stChatMessage { border-radius: 10px; padding: 12px; margin-bottom: 8px; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

agent = load_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"physics-ui-{uuid4()}"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Physics Study Buddy")
    st.write(
        "A capstone-grade agentic AI assistant for B.Tech physics students. "
        "It answers from a grounded physics knowledge base, remembers thread context, "
        "uses tools for date/time and arithmetic, and evaluates answer faithfulness."
    )
    st.subheader("Explore Knowledge Base")
    categories = ["All Categories"] + agent.knowledge_base.get_all_categories()
    selected_category = st.selectbox("Topic Category", categories)
    with st.expander("📖 Category Document Breakdown"):
        stats = agent.knowledge_base.get_category_stats()
        for cat, count in stats.items():
            st.write(f"- **{cat}**: {count} topics")

    with st.expander("🔍 Direct Keyword Search"):
        kw_query = st.text_input("Search term", key="kw_search_input")
        if kw_query:
            matches = agent.knowledge_base.search_by_keyword(kw_query)
            if matches:
                st.write(f"Found **{len(matches)}** matching topic(s):")
                for m in matches:
                    st.markdown(f"- **{m['topic']}** (`{m['category']}`)")
            else:
                st.write("No matching documents found.")



    st.subheader("Formula Quick Reference")
    with st.expander("⚡ Core Physics Formulas"):
        st.markdown(
            """
            - **Kinematics**: $v = u + at$, $s = ut + \\frac{1}{2}at^2$, $v^2 = u^2 + 2as$
            - **Newton's Law**: $F = ma$, $p = mv$
            - **Work & Energy**: $W = F s \\cos\\theta$, $K = \\frac{1}{2}mv^2$, $U = mgh$
            - **Gravitation**: $F = \\frac{G m_1 m_2}{r^2}$, $g = 9.8 \\text{ m/s}^2$
            - **SHM**: $T = 2\\pi \\sqrt{\\frac{m}{k}}$, $T = 2\\pi \\sqrt{\\frac{l}{g}}$
            - **Optics**: $\\frac{1}{f} = \\frac{1}{v} - \\frac{1}{u}$, $n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2$
            - **Electrostatics**: $F = \\frac{k q_1 q_2}{r^2}$, $V = IR$, $P = VI$
            - **Thermodynamics**: $\\Delta Q = \\Delta U + \\Delta W$
            """
        )

    st.subheader("Quick Presets")

    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("Ohm's Law", use_container_width=True):
            st.session_state.preset_query = "What is Ohm's law?"
        if st.button("Calc Test", use_container_width=True):
            st.session_state.preset_query = "Calculate 12 * 4 + 5"
        if st.button("Vector Dot", use_container_width=True):
            st.session_state.preset_query = "dot product of [1, 2, 3] and [4, 5, 6]"
    with preset_col2:
        if st.button("SHM Info", use_container_width=True):
            st.session_state.preset_query = "Explain Simple Harmonic Motion"
        if st.button("Intro Name", use_container_width=True):
            st.session_state.preset_query = "My name is Nishan."
        if st.button("Convert 5km", use_container_width=True):
            st.session_state.preset_query = "Convert 5 km to m"

    st.subheader("Runtime Status")
    st.write(f"**LLM Backend:** `{agent.llm_backend.provider}`")
    st.write(f"**Embedder:** `{agent.knowledge_base.embedder_name}`")
    st.write(f"**Documents Indexed:** `{len(agent.knowledge_base.docs)}`")
    st.write(f"**Memory Window:** `{agent.max_messages_window} messages`")
    st.write(f"**Session Thread:** `{st.session_state.get('thread_id', '')[:18]}...`")
    st.write(f"**Session Turns:** `{len(st.session_state.messages) // 2}`")

    
    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    if st.session_state.messages:
        chat_export_json = json.dumps(st.session_state.messages, indent=2)
        md_lines = ["# Physics Study Buddy Chat Export\n"]
        for m in st.session_state.messages:
            role = m.get("role", "user").capitalize()
            md_lines.append(f"### {role}\n{m.get('content', '')}\n")
        chat_export_md = "\n".join(md_lines)

        st.download_button(
            label="📥 Export Chat History (JSON)",
            data=chat_export_json,
            file_name="physics_study_buddy_chat.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            label="📄 Export Chat History (Markdown)",
            data=chat_export_md,
            file_name="physics_study_buddy_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )


st.title("Physics Study Buddy")
st.caption("Ask concept questions, formulas, memory follow-ups, or simple calculations.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your physics question")
if "preset_query" in st.session_state and st.session_state.preset_query:
    prompt = st.session_state.pop("preset_query")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = agent.ask(prompt, st.session_state.thread_id)
    raw_answer = result.get("answer", "I could not produce an answer.")
    formatted_answer = format_physics_formulas(raw_answer)
    route_name = result.get("route", "unknown")
    faithfulness_score = result.get("faithfulness", 0.0)
    elapsed_time = result.get("elapsed_seconds", 0.0)
    sources = result.get("sources", [])

    full_response = formatted_answer
    st.session_state.messages.append({"role": "assistant", "content": full_response, "route": route_name, "faithfulness": faithfulness_score, "sources": sources, "elapsed": elapsed_time})
    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.caption(f"Route: `{route_name}` | Faithfulness: `{faithfulness_score:.2f}` | Latency: `{elapsed_time:.3f}s`")
        if sources:
            with st.expander("📚 View Grounded Sources"):
                for src in sources:
                    st.write(f"- **{src.get('topic', 'Topic')}** (distance: {src.get('distance', 0.0)})")


