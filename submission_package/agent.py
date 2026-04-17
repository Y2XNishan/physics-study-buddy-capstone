from __future__ import annotations

import argparse
import json

from physics_study_buddy.agent_core import build_agent, default_thread_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics Study Buddy capstone agent")
    parser.add_argument("--question", type=str, help="Question to ask the agent")
    parser.add_argument("--thread-id", type=str, default=default_thread_id())
    args = parser.parse_args()

    agent = build_agent()
    if not args.question:
        print("Provide a question with --question to query the agent.")
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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

