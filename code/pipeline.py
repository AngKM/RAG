from code.model.model import Agent

def run():
    agent = Agent()
    print(agent.run("What is the capital of France?").choices[0].message.content)