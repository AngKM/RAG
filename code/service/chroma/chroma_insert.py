from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from config import DATA_NAME
#This class is for inputting data into the vector database
class ChromaInsert:
    def __init__(self):
        self.data_name = DATA_NAME
        self.client = chromadb.PersistentClient(path="./files/"+self.data_name) 
        self.collection = self.client.get_or_create_collection(name=self.data_name)
    
    def add_data(self, raw_data_path):
        with open(raw_data_path, "r", encoding="utf-8") as f:
            content = f.read()
        count_before = self.collection.count()
        print("Count Before: ", count_before)
        #Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(content)
        documents = []
        metadata = []
        ids = []
        for i, chunk in enumerate(chunks):
            # print("Chunk Apeending: ", chunk)
            documents.append(chunk)
            metadata.append({"source": "startup_guide"})
            ids.append(str(i))
        
        self.collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        count_after = self.collection.count()
        print("Count After: ", count_after)
        print("Total Data Added: ", count_after - count_before)
        print("Data added successfully") if count_after > count_before else print("Data not added")
if __name__ == "__main__":
    chroma = ChromaInsert()
    chroma.add_data("files/startup_guide/startup_guide.txt")
    