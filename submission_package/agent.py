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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--top-k", type=int, default=3, help="Top K documents to retrieve (default: 3)")
    parser.add_argument("--version", "-v", action="version", version="Physics Study Buddy Agent v1.0.0 (Capstone Submission)")
    args = parser.parse_args()


    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    agent = build_agent(top_k=args.top_k)


    if args.interactive:
        print("=== Physics Study Buddy Interactive Chat ===")
        print(f"Thread ID: {args.thread_id}")
        print("Commands: 'exit' or 'quit' to end session | '/reset' to clear conversation memory.\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit"}:
                    print("Goodbye! Thanks for using Physics Study Buddy.")
                    break
                if user_input.lower() in {"/reset", "reset", "clear"}:
                    agent.reset_thread(args.thread_id)
                    print("[Memory reset for current thread.]\n")
                    continue
                result = agent.ask(user_input, args.thread_id)
                print(f"\nAssistant: {result.get('answer', '')}")
                print(f"[Route: {result.get('route')} | Faithfulness: {result.get('faithfulness')}]\n")
            except (KeyboardInterrupt, EOFError):
                print("\n[Session terminated by user. Goodbye!]")
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

