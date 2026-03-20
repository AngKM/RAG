import chromadb
from config import DATA_NAME

class ChromaRetrieve:
    def __init__(self):
        self.collection = chromadb.PersistentClient(path="./files/"+DATA_NAME).get_or_create_collection(name=DATA_NAME)
        print("Retrieving data from:", "./files/"+DATA_NAME)
    
    def retrieve(self, query):
        return self.collection.query(
            query_texts=[query],
            n_results=100
        )