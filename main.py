import os
from flask import Flask, request, jsonify
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

app = Flask(__name__)

# Initialize the Agent
web_agent = Agent(
    name="Web Agent",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get("question", "What's happening?")
    response = web_agent.print_response(question, stream=False)
    return jsonify({"response": response})

# Main entry point for running the app
if __name__ == '__main__':
    # Render requires the app to bind to 0.0.0.0 and the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
