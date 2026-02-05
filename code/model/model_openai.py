from openai import OpenAI
from dotenv import load_dotenv

class Agent:
    def __init__(self):
        load_dotenv(override=True)
        self.client = OpenAI()
    
    def get_all_response(self, query):
        return self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
        )
    def get_response(self, query):
        return self.get_all_response(query).choices[0].message.content