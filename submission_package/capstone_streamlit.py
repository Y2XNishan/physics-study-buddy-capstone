from __future__ import annotations

from uuid import uuid4

import streamlit as st

from physics_study_buddy.agent_core import build_agent


@st.cache_resource
def load_agent():
    return build_agent()


def reset_conversation() -> None:
    if "thread_id" in st.session_state and "agent" in globals():
        load_agent().reset_thread(st.session_state.thread_id)
    st.session_state.thread_id = f"physics-ui-{uuid4()}"
    st.session_state.messages = []


st.set_page_config(page_title="Physics Study Buddy", page_icon="P", layout="wide")

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
    st.subheader("Topics Covered")
    for topic in agent.knowledge_base.topics:
        st.write(f"- {topic}")
    st.subheader("Runtime")
    st.write(f"LLM backend: `{agent.llm_backend.provider}`")
    st.write(f"Embedding backend: `{agent.knowledge_base.embedder_name}`")
    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

st.title("Physics Study Buddy")
st.caption("Ask concept questions, formulas, memory follow-ups, or simple calculations.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your physics question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = agent.ask(prompt, st.session_state.thread_id)
    answer = result.get("answer", "I could not produce an answer.")
    meta = (
        f"\n\nRoute: `{result.get('route', 'unknown')}`"
        f" | Faithfulness: `{result.get('faithfulness', 0.0)}`"
    )
    full_answer = answer + meta
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    with st.chat_message("assistant"):
        st.markdown(full_answer)

