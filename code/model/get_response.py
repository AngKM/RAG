#This is for choosing which company to use for the response
from config import MODEL_COMPANY, MODEL_NAME

if MODEL_COMPANY == "OpenAI":
    from code.model.OpenAI.get_OpenAI_response import OpenAI_Agent
    agent = OpenAI_Agent(model=MODEL_NAME)
elif MODEL_COMPANY == "Anthropic":
    from code.model.Anthropic.get_Anthropic_response import Anthropic_Agent
    agent = Anthropic_Agent(model=MODEL_NAME)
else:
    raise ValueError("Invalid model name")

def get_response(query, system_prompt):
    return agent.get_response(query, system_prompt)

