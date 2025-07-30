import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict

from langchain_core.runnables import RunnableConfig


DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report
   
Provide a paragraph with no more than 500 words to describe the key take aways on the topic"""


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"


class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE  # Defaults to the default report structure

    ### SET THESE NUMBERS HIGHER FOR A LARGER / MORE DETAILED REPORT - YOU MAY RUN INTO RATE LIMITING ISSUES
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations

    ### UNCOMMENT BELOW IF YOU RUN INTO RATE LIMIT ISSUES
    # planner_provider: PlannerProvider = PlannerProvider.OPENAI  # Defaults to OpenAI as provider
    # planner_model: str = "gpt-4o-mini" # Defaults to gpt-4o-mini
    # writer_provider: WriterProvider = WriterProvider.OPENAI # Defaults to OpenAI as provider
    # writer_model: str = "gpt-4o-mini" # Defaults to gpt-4o-mini

    ### COMMENT BELOW IF YOU RUN INTO RATE LIMIT ISSUES
    planner_provider: PlannerProvider = PlannerProvider.ANTHROPIC  # Defaults to Anthropic as provider
    planner_model: str = "claude-3-5-sonnet-20240620"  # Defaults to claude-3-5-sonnet-20240620
    writer_provider: WriterProvider = WriterProvider.ANTHROPIC  # Defaults to Anthropic as provider
    writer_model: str = "claude-3-5-sonnet-20240620"  # Defaults to claude-3-5-sonnet-20240620

    search_api: SearchAPI = SearchAPI.TAVILY  # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v}) 