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


def test_calculator_math_functions_and_division_by_zero():
    from physics_study_buddy.tools import calculate_expression
    
    # Test sqrt and functions
    res_sqrt = calculate_expression("Calculate sqrt(16) + 5")
    assert "9.0000" in res_sqrt

    # Test division by zero
    res_zero = calculate_expression("Calculate 10 / 0")
    assert "Division by zero is undefined" in res_zero

    # Test overflow
    res_ovf = calculate_expression("Calculate 10 ** 1000")
    assert "overflow" in res_ovf or "error" in res_ovf

