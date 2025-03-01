import os
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import time
import re
import json

from flask import Flask, request, jsonify, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from werkzeug.middleware.proxy_fix import ProxyFix

class Config:
    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 3
    RATE_LIMIT = "100 per day"
    RATE_LIMIT_BURST = "5 per minute"
    CACHE_DURATION = 3600
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    MAX_SEARCH_RESULTS = 10

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
        """Initialize search tool separately."""
        self.search_tool = DuckDuckGo()

    def initialize_agent(self) -> None:
        try:
            self.web_agent = Agent(
                name="Enhanced Web Research Agent",
                model=Gemini(id="gemini-2.0-flash-exp"),
                tools=[self.search_tool],
                instructions=[
                    "You are a real-time web research assistant.",
                    "Analyze and synthesize information from multiple sources.",
                    "Always cite sources with URLs and dates.",
                    "Present information in clear, structured markdown.",
                    "Focus on recent and reliable sources.",
                    "Include source url.",
                ],
                show_tool_calls=True,
                markdown=True
            )
            self.logger.info("Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def format_search_context(self, search_results: str, query: str) -> str:
        """Format search results into a context string for the agent."""
        try:
            results = json.loads(search_results)
            
            context = f"Based on recent search results for '{query}', here's what I found:\n\n"
            
            # Add each search result to the context
            for result in results:
                if result.get('body') and result.get('title'):
                    context += f"Source: {result.get('title')}\n"
                    if result.get('date'):
                        context += f"Date: {result.get('date')}\n"
                    context += f"URL: {result.get('link', 'No URL available')}\n"
                    context += f"Content: {result.get('body')}\n\n"
            
            context += "\nPlease analyze these sources and provide a comprehensive summary that:\n"
            context += "1. Focuses on the most recent information\n"
            context += "2. Includes specific dates and details\n"
            context += "3. Cites sources for key claims\n"
            context += "4. Notes any contradictions between sources\n"
            
            return context
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse search results")
            return f"Please analyze the following query and provide recent information: {query}"

    async def get_response_with_search(self, question: str) -> str:
        """Get response using explicit search and processing."""
        try:
            # First, perform the search
            self.logger.info(f"Searching for: {question}")
            search_results = self.search_tool.duckduckgo_search(
                query=question,
                max_results=Config.MAX_SEARCH_RESULTS
            )

            if not search_results:
                return "No search results found for your query."

            # Format search results into context
            context = self.format_search_context(search_results, question)

            # Get response from agent with search context
            response = self.web_agent.run(
                message=context,
                stream=False
            )

            return self.extract_response_content(response)

        except Exception as e:
            self.logger.error(f"Error in search and response: {str(e)}")
            raise

    def extract_response_content(self, response: Any) -> str:
        """Extract content from response object."""
        try:
            if hasattr(response, 'content'):
                return str(response.content)
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            self.logger.error(f"Error extracting response content: {str(e)}")
            return "Error processing response"

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

                # Perform search and get response
                search_results = self.search_tool.duckduckgo_search(
                    query=question,
                    max_results=Config.MAX_SEARCH_RESULTS
                )
                
                if not search_results:
                    return jsonify({"error": "No search results found"}), 404

                # Format context and get response
                context = self.format_search_context(search_results, question)
                response = self.web_agent.run(message=context, stream=False)
                
                return jsonify({
                    "response": self.extract_response_content(response),
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "processed_at": time.time(),
                        "source": "gemini-2.o-flash-exp",
                        "search_results_count": len(json.loads(search_results))
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
        raise

if __name__ == '__main__':
    main()
