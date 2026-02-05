from openai import OpenAI
from dotenv import load_dotenv

class Agent:
    def __init__(self):
        load_dotenv(override=True)
        self.client = OpenAI()
    
    def run(self, query):
        return self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
        )