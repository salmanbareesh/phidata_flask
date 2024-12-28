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

# Serverless function entry point
def handler(request):
    with app.request_context(request):
        return app.full_dispatch_request()
