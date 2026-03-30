from openai import OpenAI
from dotenv import load_dotenv

class OpenAI_Agent:
    def __init__(self, model):
        load_dotenv(override=True)
        self.client = OpenAI()
        self.model = model
    
    def get_all_response(self, query, system_prompt):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=1024,
            top_p=0.9,
        )
    def get_response(self, query, system_prompt):
        return self.get_all_response(query, system_prompt).choices[0].message.content

if __name__ == "__main__":
    agent = OpenAI_Agent("gpt-4o")
    print(agent.get_response("Hello", "You are a helpful assistant"))