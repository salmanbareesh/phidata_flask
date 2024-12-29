import os
from flask import Flask, request, jsonify
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

app = Flask(__name__)

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
    try:
        # Parse the incoming JSON data
        data = request.json
        question = data.get("question", "What's happening?")
        print(f"Received question: {question}")  # Debugging line
        
        # Call the agent's method to get the response
        response = web_agent.print_response(question, stream=False)
        print(f"Generated response: {response}")  # Debugging line
        
        if response:
            # Ensure the response is returned as JSON
            return jsonify({"response": response})
        else:
            # Return a fallback message if no response is generated
            return jsonify({"response": "No response generated"})
    
    except Exception as e:
        # Catch and log any errors
        print(f"Error processing request: {e}")
        return jsonify({"response": "An error occurred"})

if __name__ == '__main__':
    # Bind to all IP addresses and use the port specified by the environment (default to 5000)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
