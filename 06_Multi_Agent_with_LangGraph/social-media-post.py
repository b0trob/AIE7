#!/usr/bin/env python3
"""
Fixed version of Social Media Post Generation Graph
This version ensures agents actually use the tools to create files
"""

import os
import getpass
import functools
import operator
from typing import Any, Callable, List, Optional, TypedDict, Union, Annotated, Dict
from pathlib import Path
import uuid

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

# ArXiv API
import arxiv
import requests
from datetime import datetime

# Set up working directory
def create_random_subdirectory():
    random_id = str(uuid.uuid4())[:8]
    indentifier = "social-media-post"
    subdirectory_path = os.path.join('./content/data', indentifier, random_id)
    os.makedirs(subdirectory_path, exist_ok=True)
    return subdirectory_path

WORKING_DIRECTORY = Path(create_random_subdirectory())

# Global variable to store paper data
current_paper_data = {}

# Tools for Research Team
@tool
def fetch_paper_details(
    paper_id: Annotated[str, "ArXiv ID or paper identifier (e.g., '1706.03762' for 'Attention Is All You Need')"]
) -> str:
    """Fetch comprehensive details about a specific ML paper from ArXiv."""
    try:
        # Search for the paper
        search = arxiv.Search(id_list=[paper_id])
        results = list(search.results())
        
        if not results:
            return f"Paper with ID {paper_id} not found on ArXiv."
        
        paper = results[0]
        
        # Extract paper details
        paper_info = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published_date": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
            "arxiv_id": paper.entry_id,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi if hasattr(paper, 'doi') else None
        }
        
        # Store in global state for other tools to access
        global current_paper_data
        current_paper_data = paper_info
        
        return f"""Paper Details:
Title: {paper_info['title']}
Authors: {', '.join(paper_info['authors'])}
Published: {paper_info['published_date']}
ArXiv ID: {paper_info['arxiv_id']}
Categories: {', '.join(paper_info['categories'])}
DOI: {paper_info['doi']}

Abstract:
{paper_info['abstract']}

PDF URL: {paper_info['pdf_url']}"""
        
    except Exception as e:
        return f"Error fetching paper details: {str(e)}"

@tool
def search_related_work(
    query: Annotated[str, "Search query for related papers"]
) -> str:
    """Search for related papers and research in the same domain."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
        
        related_papers = []
        for paper in results:
            related_papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "arxiv_id": paper.entry_id
            })
        
        if not related_papers:
            return "No related papers found."
        
        result_text = "Related Papers:\n\n"
        for i, paper in enumerate(related_papers, 1):
            result_text += f"{i}. {paper['title']}\n"
            result_text += f"   Authors: {', '.join(paper['authors'])}\n"
            result_text += f"   Published: {paper['published']}\n"
            result_text += f"   ArXiv ID: {paper['arxiv_id']}\n\n"
        
        return result_text
        
    except Exception as e:
        return f"Error searching related work: {str(e)}"

# Tools for Content Creation Team
@tool
def create_post_draft(
    research_summary: Annotated[str, "Research summary from research team"],
    target_audience: Annotated[str, "Target LinkedIn audience"]
) -> str:
    """Create initial LinkedIn post draft based on research findings."""
    global current_paper_data
    
    if not current_paper_data:
        return "No paper data available. Please fetch paper details first."
    
    # Create a professional LinkedIn post draft
    post_draft = f"""🔬 New Research Alert: {current_paper_data['title']}

📄 Paper: {current_paper_data['title']}
👥 Authors: {', '.join(current_paper_data['authors'])}
📅 Published: {current_paper_data['published_date']}
🔗 ArXiv: {current_paper_data['arxiv_id']}

💡 Key Insights:
{research_summary}

🎯 Why This Matters:
This research represents a significant advancement in the field, offering practical applications for {target_audience}.

📊 Impact:
• Novel methodology that could revolutionize current approaches
• Potential applications across multiple domains
• Strong empirical validation of the proposed techniques

🔍 Takeaway:
This paper demonstrates the importance of [key finding] and its implications for future research and applications.

#MachineLearning #AI #Research #AcademicTwitter #DataScience #Innovation #TechResearch

What are your thoughts on this research? Share your insights below! 👇

[Link to paper: {current_paper_data['pdf_url']}]"""
    
    # Save to file
    file_path = WORKING_DIRECTORY / "linkedin_post_draft.txt"
    with open(file_path, "w") as f:
        f.write(post_draft)
    
    return f"LinkedIn post draft created and saved to {file_path}\n\nDraft:\n{post_draft}"

@tool
def optimize_for_linkedin(
    content: Annotated[str, "Post content to optimize"]
) -> str:
    """Optimize content specifically for LinkedIn platform and audience."""
    # LinkedIn optimization rules
    optimized_content = content
    
    # Ensure professional tone
    optimized_content = optimized_content.replace("🔥", "💡")
    optimized_content = optimized_content.replace("🚀", "📈")
    
    # Add professional hashtags if not present
    if "#MachineLearning" not in optimized_content:
        optimized_content += "\n\n#MachineLearning #AI #Research #DataScience"
    
    # Ensure proper formatting for LinkedIn
    optimized_content = optimized_content.replace("\n\n", "\n\n")
    
    # Save optimized version
    file_path = WORKING_DIRECTORY / "linkedin_post_optimized.txt"
    with open(file_path, "w") as f:
        f.write(optimized_content)
    
    return f"Content optimized for LinkedIn and saved to {file_path}\n\nOptimized content:\n{optimized_content}"

@tool
def add_hashtags_and_mentions(
    content: Annotated[str, "Post content"],
    paper_authors: Annotated[str, "Paper authors to mention"]
) -> str:
    """Add relevant hashtags and author mentions to the post."""
    # Add author mentions if they have LinkedIn profiles
    # In a real implementation, you would look up LinkedIn profiles
    enhanced_content = content
    
    # Add relevant hashtags based on paper content
    hashtags = [
        "#MachineLearning", "#AI", "#Research", "#DataScience", 
        "#Innovation", "#TechResearch", "#AcademicTwitter"
    ]
    
    # Add domain-specific hashtags based on paper categories
    global current_paper_data
    if current_paper_data and 'categories' in current_paper_data:
        categories = current_paper_data['categories']
        if 'cs.AI' in categories:
            hashtags.append("#ArtificialIntelligence")
        if 'cs.LG' in categories:
            hashtags.append("#DeepLearning")
        if 'cs.CL' in categories:
            hashtags.append("#NLP")
        if 'cs.CV' in categories:
            hashtags.append("#ComputerVision")
    
    # Add hashtags to content
    if not any(hashtag in enhanced_content for hashtag in hashtags):
        enhanced_content += f"\n\n{' '.join(hashtags)}"
    
    # Save enhanced version
    file_path = WORKING_DIRECTORY / "linkedin_post_final.txt"
    with open(file_path, "w") as f:
        f.write(enhanced_content)
    
    return f"Hashtags and mentions added. Saved to {file_path}\n\nEnhanced content:\n{enhanced_content}"

def create_agent(llm, tools, system_prompt):
    """Create a function-calling agent with specified tools and prompt."""
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason!")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def main():
    """Main function to run the social media post generation system."""
    print("🔬 Social Media Post Generation System (Fixed Version)")
    print("=" * 50)
    
    # Get API keys
    print("\n📝 Please provide your API keys:")
    openai_key = getpass.getpass("OpenAI API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Get paper information
    print("\n📄 Paper Information:")
    paper_id = input("Enter ArXiv ID (e.g., '1706.03762' for 'Attention Is All You Need'): ").strip()
    
    if not paper_id:
        print("❌ No paper ID provided. Exiting.")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    print(f"\n🚀 Starting post generation for paper: {paper_id}")
    print("=" * 50)
    
    try:
        # Step 1: Fetch paper details directly
        print("\n1️⃣ Fetching paper details...")
        paper_details = fetch_paper_details(paper_id)
        print(paper_details)
        
        if "not found" in paper_details.lower():
            print("❌ Paper not found. Exiting.")
            return
        
        # Step 2: Create LinkedIn post using agent
        print("\n2️⃣ Creating LinkedIn post...")
        content_writer_agent = create_agent(
            llm, 
            [create_post_draft], 
            "You are an expert at creating engaging LinkedIn posts about technical topics. You MUST use the create_post_draft tool to write the post to a file. Always use the available tools to complete your tasks."
        )
        
        # Create a simple input for the agent
        agent_input = {
            "messages": [HumanMessage(content=f"Create a LinkedIn post about this paper. Use the create_post_draft tool. Paper details: {paper_details[:1000]}...")]
        }
        
        result = content_writer_agent.invoke(agent_input)
        print("✅ Post draft created!")
        
        # Step 3: Optimize the post
        print("\n3️⃣ Optimizing post for LinkedIn...")
        style_editor_agent = create_agent(
            llm, 
            [optimize_for_linkedin], 
            "You are an expert at editing content for professional tone and LinkedIn optimization. You MUST use the optimize_for_linkedin tool to process content. Always use the available tools to complete your tasks."
        )
        
        # Read the draft and optimize it
        draft_file = WORKING_DIRECTORY / "linkedin_post_draft.txt"
        if draft_file.exists():
            with open(draft_file, "r") as f:
                draft_content = f.read()
            
            optimize_input = {
                "messages": [HumanMessage(content=f"Optimize this LinkedIn post for professional tone and LinkedIn platform. Use the optimize_for_linkedin tool. Post content: {draft_content}")]
            }
            
            optimize_result = style_editor_agent.invoke(optimize_input)
            print("✅ Post optimized!")
        
        # Step 4: Add hashtags and mentions
        print("\n4️⃣ Adding hashtags and mentions...")
        engagement_agent = create_agent(
            llm, 
            [add_hashtags_and_mentions], 
            "You are an expert at optimizing social media content for engagement. You MUST use the add_hashtags_and_mentions tool. Always use the available tools to complete your tasks."
        )
        
        # Read the optimized content and add hashtags
        optimized_file = WORKING_DIRECTORY / "linkedin_post_optimized.txt"
        if optimized_file.exists():
            with open(optimized_file, "r") as f:
                optimized_content = f.read()
            
            hashtag_input = {
                "messages": [HumanMessage(content=f"Add relevant hashtags and author mentions to this LinkedIn post. Use the add_hashtags_and_mentions tool. Post content: {optimized_content}")]
            }
            
            hashtag_result = engagement_agent.invoke(hashtag_input)
            print("✅ Hashtags and mentions added!")
        
        print("\n✅ All steps completed!")
        print("=" * 50)
        
        # Display the final post
        print("\n📝 FINAL LINKEDIN POST:")
        print("=" * 50)
        
        # Try to read the final post from the working directory
        final_post_file = WORKING_DIRECTORY / "linkedin_post_final.txt"
        if final_post_file.exists():
            with open(final_post_file, "r") as f:
                final_post = f.read()
            print(final_post)
        else:
            # Fallback to reading the optimized version
            optimized_post_file = WORKING_DIRECTORY / "linkedin_post_optimized.txt"
            if optimized_post_file.exists():
                with open(optimized_post_file, "r") as f:
                    final_post = f.read()
                print(final_post)
            else:
                # Fallback to reading the draft
                draft_post_file = WORKING_DIRECTORY / "linkedin_post_draft.txt"
                if draft_post_file.exists():
                    with open(draft_post_file, "r") as f:
                        final_post = f.read()
                    print(final_post)
                else:
                    print("❌ No post file found. Check the working directory for output files.")
        
        print(f"\n📁 Files saved in: {WORKING_DIRECTORY}")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 