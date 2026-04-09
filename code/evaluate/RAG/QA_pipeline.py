from code.evaluate.RAG.QGAC.question import Question
from code.evaluate.RAG.QGAC.answer import Answer

class QAPipeline:
    def __init__(self):
        self.question = Question()
        self.answer = Answer(self.question.QA_data)
    
    def run(self):
        return self.answer.generate_answer()

if __name__ == "__main__":
    qa_pipeline = QAPipeline()
    qa_pipeline.run()