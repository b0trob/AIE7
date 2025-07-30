import os
import asyncio
import requests
from tavily import TavilyClient, AsyncTavilyClient
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from exa_py import Exa
from typing import List, Optional, Dict, Any
from langsmith import traceable

from .state import Section

tavily_client = TavilyClient()
tavily_async_client = AsyncTavilyClient()


def get_config_value(value):
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value


def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],
        "perplexity": [],
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
    }

    accepted_params = SEARCH_API_PARAMS.get(search_api, [])
    if not search_api_config:
        return {}
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source.
    """
    sources_list = []
    for response in search_response:
        if 'results' in response:
            sources_list.extend(response['results'])

    unique_sources = {source['url']: source for source in sources_list if 'url' in source}

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source.get('title', 'No Title')}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', '')}\n===\n"
        if include_raw_content:
            char_limit = max_tokens_per_source * 4
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}
"""
    return formatted_str


@traceable
async def tavily_search_async(search_queries):
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=5,
                include_raw_content=True,
                topic="general"
            )
        )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs


@traceable
def perplexity_search(search_queries):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    search_docs = []
    for query in search_queries:
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources."
                },
                {"role": "user", "content": query},
            ],
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        results = []
        results.append({
            "title": f"Perplexity Search, Source 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0
        })
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",
                "url": citation,
                "content": "See primary source for full content",
                "raw_content": None,
                "score": 0.5
            })
        search_docs.append({
            "query": query,
            "results": results
        })
    return search_docs


@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5,
                     include_domains: Optional[List[str]] = None,
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")

    exa = Exa(api_key=f"{os.getenv('EXA_API_KEY')}")

    async def process_query(query):
        loop = asyncio.get_event_loop()

        def exa_search_fn():
            kwargs = {
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,
                "num_results": num_results
            }
            if subpages is not None:
                kwargs["subpages"] = subpages
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            return exa.search_and_contents(query, **kwargs)

        response = await loop.run_in_executor(None, exa_search_fn)
        formatted_results = []
        seen_urls = set()

        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default

        results_list = get_value(response, 'results', [])
        for result in results_list:
            score = get_value(result, 'score', 0.0)
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            content = f"{summary_content}\n\n{text_content}".strip()
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            if url in seen_urls:
                continue
            seen_urls.add(url)
            formatted_results.append({
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            })

        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    subpage_score = get_value(subpage, 'score', 0.0)
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    subpage_content = f"{subpage_summary}\n\n{subpage_text}".strip()
                    subpage_url = get_value(subpage, 'url', '')
                    if subpage_url in seen_urls:
                        continue
                    seen_urls.add(subpage_url)
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })

        return {
            "query": query,
            "results": formatted_results
        }

    search_docs = []
    for i, query in enumerate(search_queries):
        if i > 0:
            await asyncio.sleep(0.25)
        try:
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            search_docs.append({"query": query, "results": [], "error": str(e)})
            if "429" in str(e):
                await asyncio.sleep(1.0)
    return search_docs


@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    async def process_single_query(query):
        try:
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            results = []
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                url = metadata.get('entry_id', '')
                content_parts = [f"Summary: {metadata.get('Summary', '')}"]
                # Add other metadata fields
                content = "\n".join(content_parts)
                results.append({
                    'title': metadata.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                })
            return {'query': query, 'results': results}
        except Exception as e:
            return {'query': query, 'results': [], 'error': str(e)}

    search_docs = []
    for i, query in enumerate(search_queries):
        if i > 0:
            await asyncio.sleep(3.0)
        result = await process_single_query(query)
        search_docs.append(result)
    return search_docs


@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    async def process_single_query(query):
        try:
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            results = []
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            for i, doc in enumerate(docs):
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                content_parts = [f"Summary: {doc.get('Summary', '')}"]
                # Add other metadata fields
                content = "\n".join(content_parts)
                results.append({
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                })
            return {'query': query, 'results': results}
        except Exception as e:
            return {'query': query, 'results': [], 'error': str(e)}

    search_docs = []
    delay = 1.0
    for i, query in enumerate(search_queries):
        if i > 0:
            await asyncio.sleep(delay)
        result = await process_single_query(query)
        search_docs.append(result)
        if result.get('results'):
            delay = max(0.5, delay * 0.9)
        else:
            delay = min(5.0, delay * 1.5)
    return search_docs 