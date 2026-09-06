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

The project includes ten domain questions, two red-team prompt-injection tests, multi-turn memory verification, physical constants and scientific notation calculator tests, vector math tools, degree trig functions, unit conversion tool tests, weighted phrase safety classifier tests, category filtering fallback tests, direct keyword document search tests, and 19 comprehensive pytest unit tests. All tests pass with 100% success rate:
- Faithfulness evaluation score: 1.00 average
- Answer relevancy score: 1.00 average
- Context precision score: 1.00 average
- Pytest execution time: ~0.50 seconds (19/19 tests passing)

## RAGAS Baseline / Manual Baseline

The codebase includes a baseline evaluation pipeline for grounded question-answer pairs. In offline mode, the project records baseline values for:
- Faithfulness (1.00)
- Answer relevancy (1.00)
- Context precision (1.00)

## Unique Points

- Follows the exact helper-document structure: state-first design, isolated node functions, graph routing, MemorySaver thread management, calculator/vector/datetime/unit-converter tool use, faithfulness evaluation loop, safety classifier, and Streamlit browser deployment.
- Offline-safe fallback path guarantees execution without cloud API keys.
- Robust name extraction supporting punctuation cleaning, multi-word name phrasings, and common phrase filtering.
- Math & physical constants support (`g`, `c`, `h`, `kB`, `sqrt`, `sin`, `cos`, `tan`, `sind`, `cosd`, `tand`, `asin`, `acos`, `atan`, `pi`, `e`) and scientific notation (`3e8`) with division-by-zero protection in calculator tool.
- 2D/3D Vector magnitude, dot product, and cross product functions.
- Unit conversion tool (`convert_units`) supporting length, mass, time, energy, and temperature SI conversions.
- Input safety classifier (`check_input_safety`) with weighted phrase scoring for prompt injection and out-of-scope query defense.
- Word-boundary disambiguation preventing physics terms containing 'time' (e.g. time period) from falsely triggering datetime tool.
- Category filtering with fallback, direct keyword search (`search_by_keyword`), and category document statistics in knowledge base.
- LaTeX physics equation rendering in Streamlit UI with expandable source cards, formula cheat-sheet, export history (JSON/Markdown), and quick preset question launcher buttons.
- Command-line interface with ANSI color formatting, interactive chat, `/reset` thread command, `--category`, `--export-history`, `--top-k` retriever limit, and `--version` flags.



## Future Improvements

With more time, I would add topic-wise quiz generation with answer checking and numerical problem scaffolding. A strong next step would be a formula-aware reasoning tool that can solve structured physics numericals while still grounding every step in the syllabus knowledge base and showing unit-consistent derivations.

## Project Files

- `agent.py`
- `capstone_streamlit.py`
- `run_capstone_tests.py`
- `pyproject.toml`
- `conftest.py`
- `pytest.ini`
- `physics_study_buddy/`
- `tests/test_capstone.py`
- `day13_capstone.ipynb`

## Submission Checklist

- Working project files ready
- GitHub-ready project structure ready
- Documentation draft ready
- Streamlit deployment file ready
- Agent CLI ready with interactive mode
- Pytest test suite ready with 100% pass rate
