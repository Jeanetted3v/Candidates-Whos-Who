"""To run:
python -m src.backend.main.chat_main
"""
import os
import sys
import asyncio
import logging
from typing import List
import json
from hydra import compose, initialize
import gradio as gr
from pydantic_ai import Agent


from src.backend.non_graph_db.hybrid_retrieval import HybridRetriever
from src.backend.utils.settings import SETTINGS
from src.backend.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


with initialize(version_base=None, config_path="../../../config"):
    cfg = compose(config_name="chat.yaml")


class ChatInterface:
    def __init__(self):
        """Initialize the chat interface with the hybrid retriever."""
        self.retriever = HybridRetriever(cfg)
        self.chat_history = []
        self.max_history = 5
        
    async def process_query(self, query: str, history: List[List[str]]) -> tuple:
        """Process a user query and return response with search context."""
        if not query.strip():
            return "", history
            
        logger.info(f"Processing query: {query}")
        
        # Perform search with hybrid retriever
        search_results = await self.retriever.search(query)
        logger.info(f"Search results: {search_results}")
        if not search_results:
            response = "I couldn't find any relevant information about that. Could you please rephrase your question?"
            history.append([query, response])
            return response, history
        formatted_history = ""
        if history and len(history) > 0:
            formatted_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])

        formatted_prompt=cfg.response_prompt.user_prompt.format(
            query=query,
            search_results=search_results,
            message_history=formatted_history or "No previous conversation."
        )
        logger.info(f"Formatted prompt: {formatted_prompt}")
        chat_agent = Agent(
            "openai:gpt-4.1-mini",
            result_type=str,
            system_prompt=cfg.response_prompt.system_prompt
        )
        result = await chat_agent.run(prompt=formatted_prompt)
        logger.info(f"LLM response: {result}")
        response = result.output
        logger.info(f"Response: {response}")
        
        history.append([query, response])
        # Keep only the last N interactions
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        return response, history
    
    def clear_history(self) -> List[List[str]]:
        """Clear the chat history."""
        self.chat_history = []
        return []


# Create async wrapper for Gradio
async def query_handler(query: str, history: List[List[str]]):
    # Make a deep copy of history to avoid modifying the original
    history_copy = [list(h) for h in history] if history else []
    response, _ = await interface.process_query(query, history_copy)
    history_copy.append([query, response])
    return history_copy, ""

def clear_history():
    interface.clear_history()
    return []

# Initialize the interface
interface = ChatInterface()

# Create Gradio interface
with gr.Blocks(title="Singapore GE2025 Information Assistant", 
               theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("""# Singapore Elections Information Assistant
    
Ask questions about candidates, constituencies, and political parties in Singapore's General Election 2025.
    """)
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask about candidates, constituencies, or political parties...",
            container=False,
            scale=9
        )
        submit = gr.Button("Send", scale=1)
        
    with gr.Row():
        clear = gr.Button("Clear Chat History")
        
    # Setup interactions
    msg.submit(
        fn=lambda query, history: asyncio.run(query_handler(query, history)), 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg]
    )
    
    submit.click(
        fn=lambda query, history: asyncio.run(query_handler(query, history)), 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg]
    )
    
    clear.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot]
    )


if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)