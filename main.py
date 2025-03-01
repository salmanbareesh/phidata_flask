import os
import json
import time
import logging
from datetime import datetime
from typing import Any
import httpx

from flask import Flask, request, jsonify, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from phi.agent import Agent
from phi.model.google import Gemini
from duckduckgo_search import DDGS  # Importing the DDGS class for search
from langchain.tools import Tool

class Config:
    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 3
    RATE_LIMIT = "100 per day"
    RATE_LIMIT_BURST = "5 per minute"
    CACHE_DURATION = 3600
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    MAX_SEARCH_RESULTS = 10

class CustomDuckDuckGo:
    """Custom DuckDuckGo search tool with User-Agent header to bypass rate limits."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def search(self, query: str, max_results: int = 10):
        """Perform DuckDuckGo search with a User-Agent header."""
        try:
            with httpx.Client(headers=self.headers, timeout=10) as client:
                ddgs = DDGS(client=client)
                results = list(ddgs.text(query, max_results=max_results))
                return results  # Returns a structured list, not JSON string
        except Exception as e:
            logging.error(f"Error in DuckDuckGo search: {str(e)}")
            return []

class WebAgentAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_app()
        self.setup_logging()
        self.initialize_tools()
        self.initialize_agent()

    def setup_app(self) -> None:
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1)
        CORS(self.app)
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=[Config.RATE_LIMIT],
            storage_uri="memory://"
        )

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO if not Config.DEBUG else logging.DEBUG,
            format=Config.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)

    def initialize_tools(self) -> None:
        """Initialize DuckDuckGo search tool as a valid LangChain Tool."""
        self.search_tool = Tool(
            name="DuckDuckGo Search",
            func=CustomDuckDuckGo().search,
            description="Search for real-time information using DuckDuckGo."
        )

    def initialize_agent(self) -> None:
        try:
            self.web_agent = Agent(
                name="Enhanced Web Research Agent",
                model=Gemini(id="gemini-2.0-flash-exp"),
                tools=[self.search_tool],  # Must be a valid Tool instance
                instructions=[
                    "You are a real-time web research assistant.",
                    "Analyze and synthesize information from multiple sources.",
                    "Always cite sources with URLs and dates.",
                    "Present information in clear, structured markdown.",
                    "Focus on recent and reliable sources.",
                    "Highlight any conflicting information from different sources.",
                ],
                show_tool_calls=True,
                markdown=True
            )
            self.logger.info("Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def format_search_context(self, search_results: list, query: str) -> str:
        """Format search results into a context string for the agent."""
        if not search_results:
            return f"No search results found for query: {query}"

        context = f"Based on recent search results for '{query}', here's what I found:\n\n"
        for result in search_results:
            title = result.get('title', 'No Title')
            link = result.get('href', 'No URL')
            snippet = result.get('body', 'No Content')

            context += f"**{title}**\n"
            context += f"🔗 [Link]({link})\n"
            context += f"📌 {snippet}\n\n"

        return context

    async def get_response_with_search(self, question: str) -> str:
        """Get response using explicit search and processing."""
        try:
            self.logger.info(f"Searching for: {question}")
            search_results = self.search_tool.func(
                query=question,
                max_results=Config.MAX_SEARCH_RESULTS
            )

            if not search_results:
                return "No search results found for your query."

            context = self.format_search_context(search_results, question)
            response = self.web_agent.run(message=context, stream=False)

            return self.extract_response_content(response)
        except Exception as e:
            self.logger.error(f"Error in search and response: {str(e)}")
            return "Error processing request"

    def extract_response_content(self, response: Any) -> str:
        """Extract content from response object."""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return str(response.content)
        else:
            return str(response)

    def setup_routes(self) -> None:
        @self.app.route('/query', methods=['POST'])
        @self.limiter.limit(Config.RATE_LIMIT_BURST)
        def query() -> Response:
            try:
                if not request.is_json:
                    return jsonify({"error": "Request must be JSON"}), 415

                data = request.get_json()
                question = data.get("question", "").strip()

                if not question:
                    return jsonify({"error": "Question is required"}), 400

                search_results = self.search_tool.func(
                    query=question,
                    max_results=Config.MAX_SEARCH_RESULTS
                )

                if not search_results:
                    return jsonify({"error": "No search results found"}), 404

                context = self.format_search_context(search_results, question)
                response = self.web_agent.run(message=context, stream=False)

                return jsonify({
                    "response": self.extract_response_content(response),
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "processed_at": time.time(),
                        "source": "gemini-2.0-flash-exp",
                        "search_results_count": len(search_results)
                    }
                })
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @self.app.route('/health', methods=['GET'])
        def health_check() -> Response:
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            })

    def run(self, host: str = '0.0.0.0', port: int = None) -> None:
        port = port or int(os.environ.get("PORT", 5000))
        self.setup_routes()
        self.app.run(
            host=host,
            port=port,
            debug=Config.DEBUG
        )

def main():
    try:
        api = WebAgentAPI()
        api.run()
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()
