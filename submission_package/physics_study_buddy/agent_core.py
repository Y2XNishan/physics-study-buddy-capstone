from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from physics_study_buddy.knowledge_base import KnowledgeBase, build_knowledge_base
from physics_study_buddy.llm import LLMBackend
from physics_study_buddy.tools import choose_tool


MAX_EVAL_RETRIES = 2


class CapstoneState(TypedDict, total=False):
    question: str
    messages: list[dict[str, str]]
    route: str
    retrieved: str
    sources: list[dict[str, Any]]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


@dataclass
class PhysicsStudyBuddyAgent:
    app: Any
    knowledge_base: KnowledgeBase
    llm_backend: LLMBackend

    def ask(self, question: str, thread_id: str) -> dict:
        state: CapstoneState = {"question": question}
        result = self.app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return result


def _extract_name(question: str) -> str:
    patterns = [
        r"my name is ([A-Za-z '-]{2,40})",
        r"i am ([A-Za-z '-]{2,40})",
        r"call me ([A-Za-z '-]{2,40})",
        r"name is ([A-Za-z '-]{2,40})",
    ]
    NON_NAME_WORDS = {"studying", "physics", "asking", "here", "ready", "a student", "a learner", "btech", "b.tech", "student"}
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip(".,!? ").strip()
            if raw_name and raw_name.lower() not in NON_NAME_WORDS:
                return raw_name.title()
    return ""


def build_agent() -> PhysicsStudyBuddyAgent:
    knowledge_base = build_knowledge_base()
    llm_backend = LLMBackend.from_environment()

    def memory_node(state: CapstoneState) -> CapstoneState:
        messages = list(state.get("messages", []))
        messages.append({"role": "user", "content": state.get("question", "")})
        user_name = state.get("user_name", "")
        extracted_name = _extract_name(state.get("question", ""))
        if extracted_name:
            user_name = extracted_name
        return {
            "messages": messages[-6:],
            "user_name": user_name,
        }

    def router_node(state: CapstoneState) -> CapstoneState:
        route = llm_backend.route(state.get("question", ""), state.get("messages", []))
        print(f"[TRACE] router_node -> route={route}")
        return {"route": route}

    def retrieval_node(state: CapstoneState) -> CapstoneState:
        result = knowledge_base.query(state.get("question", ""), top_k=3)
        topics = ", ".join(source["topic"] for source in result["sources"])
        print(f"[TRACE] retrieval_node -> topics={topics}")
        return {
            "retrieved": result["retrieved"],
            "sources": result["sources"],
            "tool_result": "",
        }

    def skip_retrieval_node(state: CapstoneState) -> CapstoneState:
        print("[TRACE] skip_retrieval_node -> memory_only")
        return {
            "retrieved": "",
            "sources": [],
            "tool_result": "",
        }

    def tool_node(state: CapstoneState) -> CapstoneState:
        tool_result = choose_tool(state.get("question", ""))
        print(f"[TRACE] tool_node -> {tool_result}")
        return {
            "tool_result": tool_result,
            "retrieved": "",
            "sources": [],
        }

    def answer_node(state: CapstoneState) -> CapstoneState:
        answer = llm_backend.answer(
            question=state.get("question", ""),
            retrieved=state.get("retrieved", ""),
            tool_result=state.get("tool_result", ""),
            messages=state.get("messages", []),
            eval_retries=state.get("eval_retries", 0),
            user_name=state.get("user_name", ""),
            sources=state.get("sources", []),
        )
        print(f"[TRACE] answer_node -> answer_length={len(answer)}")
        return {"answer": answer}

    def eval_node(state: CapstoneState) -> CapstoneState:
        if not state.get("retrieved"):
            print("[TRACE] eval_node -> no_retrieval_skip")
            return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0)}
        faithfulness = llm_backend.evaluate(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            retrieved=state.get("retrieved", ""),
        )
        retries = state.get("eval_retries", 0) + 1
        print(f"[TRACE] eval_node -> faithfulness={faithfulness}, retries={retries}")
        return {"faithfulness": faithfulness, "eval_retries": retries}

    def save_node(state: CapstoneState) -> CapstoneState:
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": state.get("answer", "")})
        print("[TRACE] save_node -> persisted answer")
        return {"messages": messages[-6:]}

    def route_decision(state: CapstoneState) -> str:
        return state.get("route", "retrieve")

    def eval_decision(state: CapstoneState) -> str:
        if state.get("retrieved") and state.get("faithfulness", 0.0) < 0.7:
            if state.get("eval_retries", 0) < MAX_EVAL_RETRIES:
                logger.debug("[TRACE] eval_decision -> RETRY")
                return "answer"
        logger.debug("[TRACE] eval_decision -> PASS")
        return "save"

    graph = StateGraph(CapstoneState)
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "skip": "skip",
            "tool": "tool",
        },
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "answer": "answer",
            "save": "save",
        },
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    print("Graph compiled successfully")
    return PhysicsStudyBuddyAgent(app=app, knowledge_base=knowledge_base, llm_backend=llm_backend)


def default_thread_id() -> str:
    return f"physics-thread-{uuid4()}"
