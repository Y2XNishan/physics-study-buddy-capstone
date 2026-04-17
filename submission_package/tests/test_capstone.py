from physics_study_buddy.agent_core import build_agent


def test_graph_answers_physics_question():
    agent = build_agent()
    result = agent.ask("What is Ohm's law?", "pytest-ohm")
    assert result["route"] == "retrieve"
    assert "V = IR" in result["answer"] or "ohm" in result["answer"].lower()


def test_memory_persists_name():
    agent = build_agent()
    thread_id = "pytest-memory"
    agent.ask("My name is Ria.", thread_id)
    result = agent.ask("What is my name?", thread_id)
    assert "Ria" in result["answer"]


def test_tool_route_for_calculator():
    agent = build_agent()
    result = agent.ask("Calculate 12 / 3 + 2", "pytest-tool")
    assert result["route"] == "tool"
    assert "Calculator result" in result["answer"]

