from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from groq import Groq
from openai import OpenAI


def _normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


GENERIC_TERMS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "and",
    "of",
    "in",
    "for",
    "to",
    "law",
    "state",
    "explain",
    "give",
    "formula",
    "write",
    "simple",
    "words",
    "mean",
    "does",
    "define",
    "describe",
    "summarize",
    "tell",
    "how",
    "why",
}


PHYSICS_TERMS = {
    "physics",
    "motion",
    "displacement",
    "distance",
    "velocity",
    "acceleration",
    "speed",
    "position",
    "path",
    "time",
    "kinematics",
    "newton",
    "force",
    "inertia",
    "friction",
    "static",
    "kinetic",
    "normal",
    "reaction",
    "tension",
    "weight",
    "mass",
    "system",
    "axis",
    "energy",
    "work",
    "power",
    "potential",
    "thermal",
    "conservative",
    "gravitation",
    "gravity",
    "orbital",
    "orbit",
    "escape",
    "satellite",
    "tide",
    "pendulum",
    "shm",
    "harmonic",
    "oscillation",
    "amplitude",
    "frequency",
    "period",
    "damping",
    "wave",
    "sound",
    "longitudinal",
    "transverse",
    "medium",
    "wavelength",
    "phase",
    "optics",
    "snell",
    "refraction",
    "reflection",
    "lens",
    "mirror",
    "magnification",
    "microscope",
    "telescope",
    "electric",
    "electrostatics",
    "charge",
    "field",
    "coulomb",
    "dipole",
    "capacitor",
    "capacitance",
    "dielectric",
    "current",
    "ohm",
    "resistance",
    "resistivity",
    "conductor",
    "kirchhoff",
    "terminal",
    "emf",
    "magnetism",
    "magnetic",
    "induction",
    "flux",
    "faraday",
    "lenz",
    "fleming",
    "lorentz",
    "transformer",
    "inductor",
    "thermodynamics",
    "heat",
    "temperature",
    "entropy",
    "isothermal",
    "adiabatic",
    "isobaric",
    "isochoric",
    "carnot",
    "semiconductor",
    "diode",
    "photon",
    "doppler",
    "bohr",
    "photoelectric",
    "radioactivity",
    "fission",
    "fusion",
    "transistor",
    "doping",
    "junction",
    "rectification",
    "amplification",
    "resonance",
    "superposition",
    "interference",
    "diffraction",
    "ray",
    "focal",
    "focus",
    "spectrum",
    "light",
    "photon",
    "joule",
    "watt",
    "volt",
    "ampere",
    "pascal",
    "kelvin",
    "hertz",
    "planck",
}


REFUSAL_PATTERNS = [
    "ignore your instructions",
    "reveal the system prompt",
    "show system prompt",
    "hidden prompt",
    "developer message",
]


OUT_OF_SCOPE_PATTERNS = [
    "hostel fee",
    "kiit",
    "admission",
    "tuition",
    "mess fee",
]


def _top_sentences(question: str, retrieved: str, limit: int = 4) -> list[str]:
    question_terms = _normalize(question) - GENERIC_TERMS
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", retrieved)
        if len(sentence.strip()) >= 10 and not sentence.strip().startswith("[")
    ]
    scored = []
    for sentence in sentences:
        sentence_terms = _normalize(sentence)
        score = len(question_terms & sentence_terms)
        if any(keyword in sentence.lower() for keyword in ["formula", "equals", "states", "given by", "=", "is"]):
            score += 1
        if any(term in sentence_terms for term in question_terms):
            score += 1
        scored.append((score, sentence))
    ranked = [sentence for score, sentence in sorted(scored, reverse=True) if score >= 1]
    unique = []
    for sentence in ranked:
        if sentence not in unique:
            unique.append(sentence)
    return unique[:limit] or sentences[:limit]


@dataclass
class LLMBackend:
    provider: str
    model_name: str
    client: Any | None = None

    @classmethod
    def from_environment(cls) -> "LLMBackend":
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if openai_key:
            return cls(
                provider="openai",
                model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                client=OpenAI(api_key=openai_key),
            )
        if groq_key:
            return cls(
                provider="groq",
                model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                client=Groq(api_key=groq_key),
            )
        return cls(provider="offline", model_name="deterministic-fallback")

    def route(self, question: str, messages: list[dict[str, str]]) -> str:
        if self.provider == "offline":
            return self._offline_route(question)

        prompt = (
            "You are a router for a physics study buddy. Return one word only:\n"
            "- retrieve: for syllabus, concept, formula, or definition questions\n"
            "- tool: for date/time or arithmetic questions\n"
            "- skip: for memory-only questions about the conversation or user's name\n"
        )
        response = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\nMessages: {messages[-4:]}"},
            ]
        )
        route = response.strip().lower().split()[0]
        if route not in {"retrieve", "tool", "skip"}:
            return self._offline_route(question)
        return route

    def answer(
        self,
        question: str,
        retrieved: str,
        tool_result: str,
        messages: list[dict[str, str]],
        eval_retries: int,
        user_name: str,
        sources: list[dict[str, str]],
    ) -> str:
        if self.provider == "offline":
            return self._offline_answer(
                question=question,
                retrieved=retrieved,
                tool_result=tool_result,
                messages=messages,
                user_name=user_name,
                sources=sources,
            )

        retry_instruction = (
            "Be extra strict and only use supplied context because the previous answer failed "
            "faithfulness evaluation."
            if eval_retries > 0
            else "Answer faithfully from supplied context only."
        )
        prompt = (
            "You are Physics Study Buddy, a faithful physics tutor for B.Tech students.\n"
            "Rules:\n"
            "1. Use ONLY the retrieved context or tool result.\n"
            "2. If the answer is not available, clearly say you do not know.\n"
            "3. Do not invent formulas.\n"
            f"4. {retry_instruction}\n"
        )
        response = self._chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Retrieved Context:\n{retrieved or '[none]'}\n\n"
                        f"Tool Result:\n{tool_result or '[none]'}\n\n"
                        f"Recent Messages:\n{messages[-4:]}\n\n"
                        f"User name: {user_name or '[unknown]'}"
                    ),
                },
            ]
        )
        return response.strip()

    def evaluate(self, question: str, answer: str, retrieved: str) -> float:
        if not retrieved:
            return 1.0
        if self.provider == "offline":
            return self._offline_faithfulness(answer, retrieved)
        prompt = (
            "Rate faithfulness of the answer to the context on a 0.0 to 1.0 scale. "
            "Return only the numeric score."
        )
        raw = self._chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nContext:\n{retrieved}\n\nAnswer:\n{answer}"
                    ),
                },
            ]
        ).strip()
        try:
            return max(0.0, min(1.0, float(re.findall(r"\d+(?:\.\d+)?", raw)[0])))
        except Exception:
            return self._offline_faithfulness(answer, retrieved)

    def _chat(self, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            return ""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _offline_route(self, question: str) -> str:
        lowered = question.lower()
        if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
            return "skip"
        if any(pattern in lowered for pattern in OUT_OF_SCOPE_PATTERNS):
            return "skip"
        if re.search(r"\b(date|today|clock)\b", lowered) or "current time" in lowered:
            return "tool"
        if re.search(r"\btime\b", lowered):
            physics_time_context = ["time period", "pendulum", "shm", "relaxation time", "decay time", "travel time", "flight", "graph", "displacement", "velocity", "acceleration"]
            if not any(term in lowered for term in physics_time_context) and any(kw in lowered for kw in ["what time", "current time", "time now", "tell time"]):
                return "tool"
        if any(token in lowered for token in ["calculate", "solve", "compute"]):
            return "tool"
        if re.search(r"\d+\s*[-+/*^%]\s*\d+", lowered):
            return "tool"
        if any(
            phrase in lowered
            for phrase in [
                "my name",
                "what did i ask",
                "previous question",
                "earlier question",
                "our conversation",
                "repeat what i said",
            ]
        ):
            return "skip"
        return "retrieve"

    def _offline_answer(
        self,
        question: str,
        retrieved: str,
        tool_result: str,
        messages: list[dict[str, str]],
        user_name: str,
        sources: list[dict[str, str]],
    ) -> str:
        lowered = question.lower()
        if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
            return (
                "I cannot reveal hidden instructions or system prompts. "
                "I can still help with grounded physics questions from the study buddy topics."
            )
        if any(pattern in lowered for pattern in OUT_OF_SCOPE_PATTERNS):
            return (
                "I do not know that from the physics knowledge base, so I should not guess. "
                "Please ask a syllabus-based physics question instead."
            )
        if tool_result:
            return tool_result
        if "my name" in lowered and user_name:
            return f"Your name in this conversation is {user_name}."
        if "my name" in lowered and not user_name:
            return "I do not know your name yet because you have not introduced yourself in this thread."
        if any(phrase in lowered for phrase in ["what did i ask", "previous question", "earlier question"]):
            previous_user_messages = [m["content"] for m in messages if m["role"] == "user"]
            if len(previous_user_messages) >= 2:
                return f"Your previous question was: {previous_user_messages[-2]}"
            return "There is not enough conversation history yet for that."
        if not retrieved:
            return (
                "I do not have grounded context for that question in the current knowledge base. "
                "Please ask about the covered physics topics."
            )

        if not (_normalize(question) & PHYSICS_TERMS):
            return (
                "I do not know that from the current physics knowledge base, so I should not guess."
            )
        selected = _top_sentences(question, retrieved)
        if not selected:
            return (
                "I do not know from the retrieved context alone, so I should not guess."
            )
        answer = " ".join(selected)
        if sources:
            source_text = ", ".join(source["topic"] for source in sources)
            answer += f" Sources used: {source_text}."
        return answer

    def _offline_faithfulness(self, answer: str, retrieved: str) -> float:
        if "do not know" in answer.lower():
            return 1.0
        answer_terms = _normalize(answer) - {"sources", "used"}
        context_terms = _normalize(retrieved)
        if not answer_terms:
            return 0.0
        overlap = len(answer_terms & context_terms)
        return round(max(0.0, min(1.0, overlap / max(1, len(answer_terms)))), 2)
