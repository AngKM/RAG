from config import DATA_NAME, CRITIC_MODEL, EMBEDDING_MODEL, MODEL_NAME
import os
import json
from datasets import Dataset
import pandas as pd
import shutil

class Question:
    def __init__(self):
        self.name = DATA_NAME
        # self.QA_path = "./files/"+DATA_NAME+"/"+DATA_NAME+"_QA/"+DATA_NAME+"_QA.json"
        self.QG_name = f"QG__{DATA_NAME}__{EMBEDDING_MODEL}__{MODEL_NAME}.json"
        self.directory = f"./code/evaluate/RAG/RAG_evaluation_file/{CRITIC_MODEL}/{DATA_NAME}/{EMBEDDING_MODEL}/{MODEL_NAME}/"
        self.QA_path = f"{self.directory}/{self.QG_name}"
        self.source_path = f"./files/{DATA_NAME}/QG__{DATA_NAME}.json"
        self.ensure_path_exists() #Ensure evaluation directory exists

        self.prepare_file() #Copy the user's input into evaluation working space

        self.QA_data = self.load_data() 

        print(f"Loading Question from [{self.name}] Dataset")
        print("QA Path: ", self.QA_path)
        print("Total QA Data: ", len(self.QA_data))
    
    def ensure_path_exists(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"Directory not found, creating: {self.directory}")
        else:
            print(f"Directory found: {self.directory}")
    
    def prepare_file(self):
        """Copies the source file to the destination if it doesn't already exist."""
        # Check if the destination file already exists to avoid redundant copying
        if not os.path.exists(self.QA_path):
            if os.path.exists(self.source_path):
                print(f"Copying {self.source_path} to {self.QA_path}")
                try:
                    shutil.copy2(self.source_path, self.QA_path)
                except Exception as e:
                    print(f"Error copying file: {e}")
            else:
                print(f"No user's QG sets in {self.source_path}")
                return

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

