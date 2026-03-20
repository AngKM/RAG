class Anthropic_Agent:
    def __init__(self, model):
        print("Anthropic Agent Initialized")
    
    def get_response(self, query, system_prompt):
        print("Anthropic Agent Response")
        return "Mock Anthropic Response"

if __name__ == "__main__":
    agent = Anthropic_Agent()
    agent.get_response("Hello", "You are a helpful assistant")