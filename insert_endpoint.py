from code.service.chroma.chroma_insert import ChromaInsert
from config import DATA_NAME

class InsertEndpoint:
    def __init__(self):
        self.chroma = ChromaInsert()
    
    def insert(self):
        self.chroma.add_data("files/"+DATA_NAME+"/"+DATA_NAME+".txt")

if __name__ == "__main__":
    insert_endpoint = InsertEndpoint()
    insert_endpoint.insert()