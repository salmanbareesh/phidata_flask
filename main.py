import os
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import time
import re

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

@dataclass
class CacheEntry:
    response: str
    timestamp: float
    source_ip: str

class WebAgentAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_app()
        self.cache: Dict[str, CacheEntry] = {}
        self.setup_logging()
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

    def initialize_agent(self) -> None:
        try:
            self.web_agent = Agent(
                name="Enhanced Web Research Agent",
                model=Gemini(id="gemini-1.5-flash"),
                tools=[DuckDuckGo()],
                instructions=[
                    "Provide comprehensive and accurate information",
                    "Always cite sources with URLs when available",
                    "Format responses in clear, structured markdown",
                    "Include relevant dates and context",
                ],
                show_tool_calls=True,
                markdown=True
            )
            self.logger.info("Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def sanitize_input(self, text: str) -> str:
        text = re.sub(r'[^\w\s\-.,?!]', '', text)
        return text.strip()

    def validate_question(self, question: str) -> tuple[bool, Optional[str]]:
        if not question:
            return False, "Question cannot be empty"
        if len(question) < Config.MIN_QUESTION_LENGTH:
            return False, f"Question must be at least {Config.MIN_QUESTION_LENGTH} characters"
        if len(question) > Config.MAX_QUESTION_LENGTH:
            return False, f"Question cannot exceed {Config.MAX_QUESTION_LENGTH} characters"
        return True, None

    def extract_response_content(self, response: Any) -> str:
        """
        Extract the content from RunResponse object or return string representation.
        """
        try:
            # If response is RunResponse object, get its content
            if hasattr(response, 'content'):
                return str(response.content)
            # If response is already a string
            elif isinstance(response, str):
                return response
            # If response has a __str__ method
            else:
                return str(response)
        except Exception as e:
            self.logger.error(f"Error extracting response content: {str(e)}")
            return "Error processing response"

    def format_response(self, response: Any) -> Dict[str, Any]:
        """Format the response for JSON serialization"""
        return {
            "response": self.extract_response_content(response),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "processed_at": time.time(),
                "source": "gemini-1.5-flash",
                "cached": False
            }
        }

    def create_error_response(self, message: str, status_code: int = 400) -> Response:
        return jsonify({
            "error": {
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "status": status_code
            }
        }), status_code

    def setup_routes(self) -> None:
        @self.app.route('/query', methods=['POST'])
        @self.limiter.limit(Config.RATE_LIMIT_BURST)
        def query() -> Response:
            try:
                if not request.is_json:
                    return self.create_error_response("Request must be JSON", 415)

                data = request.get_json()
                if not isinstance(data, dict):
                    return self.create_error_response("Invalid request format", 400)

                question = data.get("question", "").strip()
                question = self.sanitize_input(question)
                is_valid, error_message = self.validate_question(question)
                
                if not is_valid:
                    return self.create_error_response(error_message, 400)

                self.logger.info(f"Processing question: {question[:50]}...")
                
                # Get response from agent
                try:
                    response = self.web_agent.run(message=question, stream=False)
                    if not response:
                        return self.create_error_response("No response generated", 500)
                    
                    # Format and return response
                    formatted_response = self.format_response(response)
                    return jsonify(formatted_response)
                
                except Exception as e:
                    self.logger.error(f"Error getting response from agent: {str(e)}")
                    return self.create_error_response("Error processing request", 500)

            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
                return self.create_error_response("Internal server error", 500)

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
