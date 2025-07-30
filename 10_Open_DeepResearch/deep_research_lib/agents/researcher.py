from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from core.state import (
    ReportState,
    SectionState,
    ReportStateInput,
    ReportStateOutput,
    SectionOutputState,
    Queries,
    Sections,
    Feedback,
)
from core.config import Configuration
from core.tools import (
    get_config_value,
    get_search_params,
    tavily_search_async,
    perplexity_search,
    exa_search,
    arxiv_search_async,
    pubmed_search_async,
    deduplicate_and_format_sources,
    format_sections,
)
from core.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    section_grader_instructions,
    final_section_writer_instructions,
)


async def generate_report_plan(state: ReportState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=get_config_value(configurable.writer_model),
        model_provider=get_config_value(configurable.writer_provider),
    ).with_structured_output(Queries)
    
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=state["topic"],
        report_organization=configurable.report_structure,
        number_of_queries=configurable.number_of_queries
    )
    
    queries = await writer_model.ainvoke([
        SystemMessage(content=system_instructions_query),
        HumanMessage(content="Generate search queries to plan the report sections.")
    ])
    
    query_list = [q.search_query for q in queries.queries]
    search_api = get_config_value(configurable.search_api)
    params_to_pass = get_search_params(search_api, configurable.search_api_config or {})
    
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
    # ... other search APIs
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    
    planner_model = init_chat_model(
        model=get_config_value(configurable.planner_model),
        model_provider=get_config_value(configurable.planner_provider),
    ).with_structured_output(Sections)
    
    system_instructions_sections = report_planner_instructions.format(
        topic=state["topic"],
        report_organization=configurable.report_structure,
        context=source_str,
        feedback=state.get("feedback_on_report_plan", "")
    )
    
    report_sections = await planner_model.ainvoke([
        SystemMessage(content=system_instructions_sections),
        HumanMessage(content="Generate the sections of the report.")
    ])
    
    return {"sections": report_sections.sections}


def human_feedback(state: ReportState, config: RunnableConfig):
    sections_str = "\n\n".join(
        f"Section: {s.name}\nDescription: {s.description}\nResearch needed: {'Yes' if s.research else 'No'}"
        for s in state['sections']
    )
    interrupt_message = f"Please provide feedback on the following report plan.\n\n{sections_str}\n\nDoes the plan meet your needs? Type 'true' to approve, or provide feedback to revise."
    feedback = interrupt(interrupt_message)
    
    if isinstance(feedback, bool) and feedback:
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": state["topic"], "section": s, "search_iterations": 0})
            for s in state['sections'] if s.research
        ])
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Unsupported feedback type: {type(feedback)}")


async def generate_queries(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=get_config_value(configurable.writer_model),
        model_provider=get_config_value(configurable.writer_provider),
    ).with_structured_output(Queries)
    
    system_instructions = query_writer_instructions.format(
        topic=state["topic"],
        section_topic=state["section"].description,
        number_of_queries=configurable.number_of_queries
    )
    
    queries = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate search queries for the section.")
    ])
    
    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    query_list = [q.search_query for q in state["search_queries"]]
    search_api = get_config_value(configurable.search_api)
    params_to_pass = get_search_params(search_api, configurable.search_api_config or {})
    
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
    # ... other search APIs
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000, include_raw_content=True)
    
    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


async def write_section(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=get_config_value(configurable.writer_model),
        model_provider=get_config_value(configurable.writer_provider),
    )
    
    system_instructions = section_writer_instructions.format(
        topic=state["topic"],
        section_name=state["section"].name,
        section_topic=state["section"].description,
        context=state["source_str"],
        section_content=state["section"].content
    )
    
    section_content = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate the report section based on the sources.")
    ])
    
    state["section"].content = section_content.content
    
    reflection_model = init_chat_model(
        model=get_config_value(configurable.planner_model),
        model_provider=get_config_value(configurable.planner_provider),
    ).with_structured_output(Feedback)
    
    grader_instructions = section_grader_instructions.format(
        topic=state["topic"],
        section_topic=state["section"].description,
        section=state["section"].content,
        number_of_follow_up_queries=configurable.number_of_queries
    )
    
    feedback = await reflection_model.ainvoke([
        SystemMessage(content=grader_instructions),
        HumanMessage(content="Grade the report section and provide follow-up queries if needed.")
    ])
    
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        return Command(update={"completed_sections": [state["section"]]}, goto=END)
    else:
        return Command(update={"search_queries": feedback.follow_up_queries, "section": state["section"]}, goto="search_web")


def gather_completed_sections(state: ReportState):
    completed_report_sections = format_sections(state["completed_sections"])
    return {"report_sections_from_research": completed_report_sections}


def initiate_final_section_writing(state: ReportState):
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]})
        for s in state['sections'] if not s.research
    ]


async def write_final_sections(state: SectionState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=get_config_value(configurable.writer_model),
        model_provider=get_config_value(configurable.writer_provider),
    )
    
    system_instructions = final_section_writer_instructions.format(
        topic=state["topic"],
        section_name=state["section"].name,
        section_topic=state["section"].description,
        context=state["report_sections_from_research"]
    )
    
    section_content = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate the final report section.")
    ])
    
    state["section"].content = section_content.content
    return {"completed_sections": [state["section"]]}


def compile_final_report(state: ReportState):
    completed_sections_map = {s.name: s.content for s in state["completed_sections"]}
    for section in state["sections"]:
        section.content = completed_sections_map.get(section.name, "")
    
    final_report = "\n\n".join([s.content for s in state["sections"]])
    return {"final_report": final_report}


def get_research_graph():
    section_builder = StateGraph(SectionState, output=SectionOutputState)
    section_builder.add_node("generate_queries", generate_queries)
    section_builder.add_node("search_web", search_web)
    section_builder.add_node("write_section", write_section)
    section_builder.add_edge(START, "generate_queries")
    section_builder.add_edge("generate_queries", "search_web")
    section_builder.add_edge("search_web", "write_section")

    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("build_section_with_web_research", section_builder.compile())
    builder.add_node("gather_completed_sections", gather_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)
    
    builder.add_edge(START, "generate_report_plan")
    builder.add_edge("generate_report_plan", "human_feedback")
    builder.add_edge("build_section_with_web_research", "gather_completed_sections")
    builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)
    
    return builder.compile() 