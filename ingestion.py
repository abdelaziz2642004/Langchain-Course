
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer




if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt",encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    print("Embeddings created")

    # # if u want locally use sentence transformer
    # model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality
    # embeddings = model.encode(texts)

    print("Creating Pinecone vector store...")
    vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Pinecone vector store created")

    # print("Ingestion complete")