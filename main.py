from flask import Flask, request, jsonify
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

# Initialize the Flask app
app = Flask(__name__)

# Initialize your Agent
web_agent = Agent(
    name="Web Agent",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Define a route for API
@app.route('/query', methods=['POST'])
def query():
    # Example: Get query from request
    data = request.json
    question = data.get("question", "What's happening?")
    
    # Get response from the agent
    response = web_agent.print_response(question, stream=False)
    return jsonify({"response": response})

# Main entry point for local testing
if __name__ == '__main__':
    app.run(debug=True)
