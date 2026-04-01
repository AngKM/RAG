from code.evaluate.test_gen import TestGen
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    with open("./files/rai_curriculum/rai_curriculum.txt", "r", encoding="utf-8") as f:
        docs = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_text(docs)

    test_gen = TestGen()
    result = test_gen.generate(docs)
    #Save the generated data to a file
    with open("./temp_generated_data.txt", "w", encoding="utf-8") as f:
        f.write(str(result))
    print("Generated data saved to temp_generated_data.txt")