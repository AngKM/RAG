from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from config import DATA_NAME, EMBEDDING_MODEL
import transformers
import re


transformers.logging.set_verbosity_error()

#This class is for inputting data into the vector database
class ChromaInsert:
    def __init__(self):
        self.data_name = DATA_NAME
        self.client = chromadb.PersistentClient(path="./files/"+self.data_name) 
        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(name=self.data_name, embedding_function=self.embedding)

    def add_data(self, raw_data_path):
        with open(raw_data_path, "r", encoding="utf-8") as f:
            content = f.read()
        #Clean Data
        content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        count_before = self.collection.count()
        print("Count Before: ", count_before)
        #Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=411,
            chunk_overlap=50,
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
            metadata.append({"source": self.data_name})
            ids.append(str(i))
        
        #Automatic embedding (default: all-MiniLM-L6-v2)
        self.collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        count_after = self.collection.count()
        print("Count After: ", count_after)
        print("Total Data Added: ", count_after - count_before)
        print("Data added successfully") if count_after > count_before else print("Data not added")

    