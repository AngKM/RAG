from config import DATA_NAME
import json
from datasets import Dataset
import pandas as pd

class Question:
    def __init__(self):
        self.name = DATA_NAME
        self.QA_path = "./files/"+DATA_NAME+"/"+DATA_NAME+"_QA/"+DATA_NAME+"_QA.json"
        self.QA_data = self.load_data()

        print(f"Loading Question from [{self.name}] Dataset")
        print("QA Path: ", self.QA_path)
        print("Total QA Data: ", len(self.QA_data))
    
    def load_data(self):
        with open(self.QA_path, "r", encoding="utf-8") as f:
            QA_data = json.load(f)
        return Dataset.from_list(QA_data)
    
    def show_data(self):
        df = pd.DataFrame(self.QA_data)
        print(df.head())

if __name__ == "__main__":
    question = Question()
    question.show_data()

