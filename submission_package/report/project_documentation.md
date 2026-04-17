# Capstone Project Documentation

## Title

Physics Study Buddy: A Faithful Agentic AI Assistant for B.Tech Physics Learners

## Mandatory Details

- Student Name: Nishan Kashyap
- Roll Number: 23053060
- Batch / Program: Agentic AI Course 2026

## Problem Statement

B.Tech students often need help revising core physics concepts outside class hours, especially when preparing for quizzes, assignments, and semester exams. Generic chatbots can sound confident while giving incorrect formulas or unsupported explanations, which makes them risky for academic use. The goal of this project is to build a grounded Physics Study Buddy that answers only from a curated physics knowledge base, remembers the current conversation through `thread_id`, uses tools when needed, and clearly admits when the answer is outside its scope.

## User

The intended users are B.Tech students who want quick, syllabus-aligned help on foundational physics topics such as motion, energy, waves, optics, electrostatics, current electricity, magnetism, thermodynamics, and modern physics.

## Solution and Features

The solution is an Agentic AI assistant implemented with LangGraph. The graph contains eight nodes: `memory`, `router`, `retrieve`, `skip`, `tool`, `answer`, `eval`, and `save`. The state is defined first using a `TypedDict`, which includes `question`, `messages`, `route`, `retrieved`, `sources`, `tool_result`, `answer`, `faithfulness`, `eval_retries`, and `user_name`.

The system uses a ChromaDB in-memory collection backed by 12 topic-specific physics documents, each written to cover one focused syllabus concept. The assistant retrieves the top three relevant documents for concept questions. A tool route handles arithmetic and current date/time questions. A memory route supports follow-up questions such as name recall or previous-question recall. Answers are evaluated for faithfulness, and low-scoring retrieval-based answers are retried up to two times.

## Tech Stack

- Python
- LangGraph
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`) with offline TF-IDF fallback
- Streamlit
- OpenAI or Groq optional LLM integration
- RAGAS-ready baseline evaluation flow with manual fallback metrics

## Architecture Summary

1. `memory_node` stores the latest user turn, applies a sliding window, and extracts the user's name if present.
2. `router_node` decides whether the question needs retrieval, tool usage, or memory-only handling.
3. `retrieval_node` queries ChromaDB and formats the top matching topic chunks.
4. `skip_retrieval_node` is used for memory-only turns.
5. `tool_node` returns calculator or date/time output and never raises exceptions.
6. `answer_node` generates a grounded answer using only retrieved context or tool output.
7. `eval_node` computes faithfulness and triggers retry when needed.
8. `save_node` appends the final assistant answer to conversation history.

## Knowledge Base Size

- 12 documents
- Topics include kinematics, Newton's laws, work-energy, gravitation, SHM, waves, optics, electrostatics, current electricity, magnetism, thermodynamics, and modern physics

## Tool Used and Why

The assistant uses two simple tools:

- Calculator: supports arithmetic questions that should not rely on retrieval
- Date/time tool: handles current date or time requests directly

These tools demonstrate routing beyond retrieval, which was a mandatory capstone requirement.

## Test Results Summary

The project includes ten domain questions, two red-team questions, and one memory test sequence. The evaluation focuses on route correctness, faithfulness score, and whether the answer stays grounded. Example red-team cases include prompt-injection attempts and out-of-scope institutional questions. The expected behavior is that the assistant does not reveal hidden instructions and does not fabricate unavailable facts.

## RAGAS Baseline / Manual Baseline

The codebase includes a baseline evaluation pipeline for five grounded question-answer pairs. If a hosted evaluation-compatible setup is available, the project can be extended to full RAGAS metrics. In offline mode, the project records manual baseline values for:

- Faithfulness
- Answer relevancy
- Context precision

These baseline values provide the initial quality checkpoint before further tuning.

## Unique Points

- Follows the exact helper-document structure: state-first design, isolated node functions, graph routing, memory, tool use, evaluation, and Streamlit deployment
- Includes an offline-safe fallback path so the project still runs without cloud API keys
- Uses focused physics documents to reduce vague retrieval behavior and minimize hallucinated formulas
- Provides both a CLI agent and a browser-based UI

## Future Improvements

With more time, I would add topic-wise quiz generation with answer checking and numerical problem scaffolding. A strong next step would be a formula-aware reasoning tool that can solve structured physics numericals while still grounding every step in the syllabus knowledge base and showing unit-consistent derivations.

## Project Files

- `agent.py`
- `capstone_streamlit.py`
- `run_capstone_tests.py`
- `physics_study_buddy/`
- `tests/test_capstone.py`
- `day13_capstone.ipynb`

## Submission Checklist

- Working project files ready
- GitHub-ready project structure ready
- Documentation draft ready
- Streamlit deployment file ready
- Agent file ready
- Test runner ready
