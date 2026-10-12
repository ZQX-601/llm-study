"""运行最小 Coordinator → Diagnostic Agent → Coordinator 链路。"""

import asyncio

try:
    from .multi_agent import build_demo_runtime, new_demo_state
except ImportError:
    from multi_agent import build_demo_runtime, new_demo_state


async def main() -> None:
    runtime, coordinator, _, backend = build_demo_runtime()
    state = await runtime.run(new_demo_state(), coordinator)
    print(f"status={state.status.value}")
    print(f"answer={state.final_answer}")
    print(f"tool_calls={len(backend.call_history)}")
    for event in state.trace:
        print(
            f"{event.sequence:02d} {event.phase:<20} "
            f"agent={event.agent_id} detail={event.detail}"
        )


if __name__ == "__main__":
    asyncio.run(main())
