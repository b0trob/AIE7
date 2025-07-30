# Deep Research Library

This library contains a reusable and configurable agent for conducting deep research on any given topic. It is built using LangGraph and is inspired by the concepts in the [Langchain Open Deep Research](https://github.com/langchain-ai/open_deep_research) project.

The agent can:
- Take a topic and autonomously plan a multi-section report.
- Conduct web research using various search APIs.
- Write content for each section, with a self-correction mechanism.
- Pause for human feedback on the initial plan.
- Compile a final report in markdown format.

## Structure

The library is organized into two main modules:

-   `core`: Contains the fundamental building blocks of the agent:
    -   `config.py`: Configuration classes for the agent (e.g., LLM providers, search APIs).
    -   `prompts.py`: All the system prompts used to instruct the LLMs.
    -   `state.py`: Pydantic and TypedDict models that define the state of the graph.
    -   `tools.py`: Search tool implementations and other utility functions.
-   `agents`: Contains the graph definition and the nodes that make up the agent's logic.
    -   `researcher.py`: Defines the LangGraph agent, its nodes, and the connections between them.

## Setup

1.  **Install Dependencies:**
    Make sure you have all the required packages installed.

    ```bash
    pip install "langgraph[llama-cpp,memgraph-and-neo4j,openai,bash,graphviz]" beautifulsoup4 markdownify pandas langchainhub langchain-community langchain-core langchain-text-splitters tavily-python "anthropic[vertex]" "openai[datalib]" "google-generativeai" "google-cloud-aiplatform>=1.38" "py-cpuinfo" "tiktoken" "lark" "langchain" "pillow" "exa_py"
    ```

2.  **Set API Keys:**
    You will need to set API keys for the LLM provider and search tools you intend to use. Create a `.env` file in the root of the project and add your keys:

    ```
    ANTHROPIC_API_KEY="your-anthropic-key"
    TAVILY_API_KEY="your-tavily-key"
    EXA_API_KEY="your-exa-key"
    PERPLEXITY_API_KEY="your-perplexity-key"
    # Optional
    OPENAI_API_KEY="your-openai-key"
    ```

## How to Use

To run the agent and generate a report, you can use the `run.py` script in the root of the `deep_research_lib` directory.

```bash
# From inside the deep_research_lib directory
python run.py
```

The script will prompt you for a research topic. For example:

```
Enter the research topic: Prompt Engineering
```

The agent will then start the research process. It will first generate a plan and present it to you for feedback.

```
Please provide feedback on the following report plan.

Section: Introduction to Prompt Engineering
Description: What is prompt engineering, its importance in AI, and its evolution.
Research needed: No

...

Does the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan:
```

If you type `true` and press Enter, the agent will proceed with the research and writing. If you provide text feedback, it will revise the plan.

The final report will be saved as `report.md` in the `deep_research_lib` directory. 