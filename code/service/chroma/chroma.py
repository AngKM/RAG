from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

class Chroma:
    def __init__(self, CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name="startup_guide")
    
    def add_data(self, raw_data_path):
        with open(raw_data_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("Count Before: ", self.collection.count())
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
            print("Chunk Apeending: ", chunk)
            documents.append(chunk)
            metadata.append({"source": "startup_guide"})
            ids.append(str(i))
        
        self.collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        print("Count After: ",self.collection.count())
        print("Data added successfully")
if __name__ == "__main__":
    chroma = Chroma("./chroma_db")
    chroma.add_data("./files/startup_guide.txt")
    