from __future__ import annotations

import argparse
import json

from physics_study_buddy.agent_core import build_agent, default_thread_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics Study Buddy capstone agent")
    parser.add_argument("--question", type=str, help="Question to ask the agent")
    parser.add_argument("--thread-id", type=str, default=default_thread_id())
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--json", action="store_true", help="Output result as formatted JSON")
    args = parser.parse_args()

    agent = build_agent()

    if args.interactive:
        print("=== Physics Study Buddy Interactive Chat ===")
        print(f"Thread ID: {args.thread_id}")
        print("Type 'exit' or 'quit' to end session.\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                result = agent.ask(user_input, args.thread_id)
                print(f"\nAssistant: {result.get('answer', '')}")
                print(f"[Route: {result.get('route')} | Faithfulness: {result.get('faithfulness')}]\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        return

    if not args.question:
        print("Provide a question with --question or use --interactive for interactive mode.")
        return

    result = agent.ask(args.question, args.thread_id)
    payload = {
        "thread_id": args.thread_id,
        "route": result.get("route"),
        "faithfulness": result.get("faithfulness"),
        "answer": result.get("answer"),
        "sources": result.get("sources"),
        "user_name": result.get("user_name"),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Answer: {payload['answer']}")
        print(f"Route: {payload['route']} | Faithfulness: {payload['faithfulness']}")


if __name__ == "__main__":
    main()

