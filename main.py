from code.model.model import Agent

agent = Agent()
print(agent.run("What is the capital of France?").choices[0].message.content)