from openai import OpenAI
from dotenv import load_dotenv

class Agent:
    def __init__(self):
        load_dotenv(override=True)
        self.client = OpenAI()
    
    def get_all_response(self, query, system_prompt):
        return self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
    def get_response(self, query, system_prompt):
        return self.get_all_response(query, system_prompt).choices[0].message.content