# Physics Study Buddy

This project implements the Agentic AI capstone in the `Study Buddy - Physics` domain from the helper document. The assistant is designed for B.Tech students who need grounded concept support outside class hours.

## What it includes

- LangGraph `StateGraph` with 8 nodes: `memory`, `router`, `retrieve`, `skip`, `tool`, `answer`, `eval`, `save`
- ChromaDB knowledge base with 12 topic-specific physics documents
- Memory persistence through `MemorySaver` and `thread_id`
- Tool support for current date/time and arithmetic
- Faithfulness evaluation with retry loop
- Streamlit deployment in `capstone_streamlit.py`
- Test runner in `run_capstone_tests.py`
- Notebook submission artifact in `day13_capstone.ipynb`
- Submission report draft in `report/project_documentation.md`

## Run locally

```bash
python -m pip install -r requirements.txt
python agent.py --question "What is Ohm's law?"
python run_capstone_tests.py
streamlit run capstone_streamlit.py
```

## Optional LLM setup

The project runs offline with a deterministic fallback. To use a hosted LLM, set either `OPENAI_API_KEY` or `GROQ_API_KEY` in your environment or a local `.env`.

## Submission files covered

- `agent.py`
- `capstone_streamlit.py`
- `day13_capstone.ipynb`
- `report/project_documentation.md`

