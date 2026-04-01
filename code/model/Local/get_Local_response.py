import ollama

class Local_Agent:
    def __init__(self, model):
        self.model = model

    def get_response(self, query, system_prompt):
        response = ollama.chat(
            model=self.model, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query},
            ], 
            options={
                'temperature': 0.0,
                "top_p": 0.9,
                "num_predict": 1024,
            }
        )

        return response['message']['content']