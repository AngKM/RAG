import chromadb
from chromadb.utils import embedding_functions 
from config import DATA_NAME, EMBEDDING_MODEL 

class ChromaRetrieve:
    def __init__(self):
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        self.client = chromadb.PersistentClient(path="./files/" + DATA_NAME)
        
        self.collection = self.client.get_or_create_collection(
            name=DATA_NAME, 
            embedding_function=self.embedding_function
        )
        print("Retrieving data from:", "./files/" + DATA_NAME)
    
    def retrieve(self, query):
        return self.collection.query(
            query_texts=[query],
            n_results=10 
        )