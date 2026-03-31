from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# The new version of 'evolutions'
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer, 
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.documents import Document
from config import GENERATOR_MODEL, CRITIC_MODEL, EMBEDDING_MODEL


class TestGen:
    def __init__(self):
        # 1. Wrap your LangChain models so Ragas can use them
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model=GENERATOR_MODEL))
        self.critic_llm = LangchainLLMWrapper(ChatOpenAI(model=CRITIC_MODEL))
        self.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
        
        # 2. Initialize the Generator
        self.generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.embeddings
        )

    def generate(self, docs):
        # 3. Use Synthesizers instead of the old 'simple', 'reasoning' constants
        processed_docs = [
            Document(page_content=d) if isinstance(d, str) else d 
            for d in docs
        ]
        # print("Processed Docs: ", processed_docs)
        #Save processed docs to file 
        with open("./temp_processed_docs.txt", "w", encoding="utf-8") as f:
            for doc in processed_docs:
                f.write(doc.page_content + "\n")
        
        run_config = RunConfig(max_workers=1, timeout=60)

        # 2. Define the distribution
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 0.5),
            # (MultiHopAbstractQuerySynthesizer(llm=self.generator_llm), 0.5)
        ]

        # 3. Use generate_with_chunks since docs are already pre-chunked
        #    by RecursiveCharacterTextSplitter. This skips the HeadlinesExtractor
        #    and HeadlineSplitter pipeline that fails on some nodes.
        return self.generator.generate_with_chunks(
            chunks=processed_docs, 
            testset_size=10, 
            query_distribution=query_distribution,
            run_config=run_config
        )