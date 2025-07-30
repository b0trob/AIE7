import asyncio
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from agents.researcher import get_research_graph
from dotenv import load_dotenv

load_dotenv()


async def run_research():
    topic = input("Enter the research topic: ")
    thread_id = str(uuid.uuid4())
    memory = MemorySaver()
    graph = get_research_graph().with_checkpoints(checkpointer=memory)

    async for chunk in graph.astream(
        {"topic": topic},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates"
    ):
        for key, value in chunk.items():
            if key == "__interrupt__":
                feedback = input(f"\n{value[0].value}\n> ")
                if feedback.lower() == 'true':
                    await resume_research(graph, thread_id, True)
                else:
                    await resume_research(graph, thread_id, feedback)
                return
            elif 'final_report' in value:
                with open("report.md", "w") as f:
                    f.write(value['final_report'])
                print("\nResearch complete. Final report saved to report.md")
                return
            else:
                print(f"--- Step: {key} ---")
                print(value)
                print("-" * 30)


async def resume_research(graph, thread_id, feedback):
    async for chunk in graph.astream(
        Command(resume=feedback),
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates"
    ):
        for key, value in chunk.items():
            if 'final_report' in value:
                with open("report.md", "w") as f:
                    f.write(value['final_report'])
                print("\nResearch complete. Final report saved to report.md")
                return
            else:
                print(f"--- Step: {key} ---")
                print(value)
                print("-" * 30)


if __name__ == "__main__":
    asyncio.run(run_research()) 