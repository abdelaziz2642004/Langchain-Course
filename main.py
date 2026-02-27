# let's start retrieval
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


if __name__ == '__main__':
    print("Retrieving...")
    embeddings=OpenAIEmbeddings()
    llm=ChatOllama(model="qwen2.5:3b")

    vector_store = PineconeVectorStore.from_existing_index(index_name=os.getenv("INDEX_NAME"),embedding=embeddings)
    retriever=vector_store.as_retriever(search_kwargs={"k":3}) # top 3 chunks relevant to my prompt

    # now we will create a prompt template
    prompt_template= ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    If the answer is not in the context, say "I don't know".
    Context: {context}
    Question: {question}
    """)

    # this function will format the docs as one string to be sent to the llm
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    def retrieval_chain_without_lcel(queryy:str):
        """
        This function will create a retrieval chain without using LCEL
        Manually retrieves documents, formats them, and generates a response

        Limitations:
        - Manual step-by-step execution
        - No built-in streaming support
        - No async support without additional code
        - Harder to compose with other chains
        - More verbose and error-prone
        """

         # Step 1: Retrieve relevant documents
        docs=retriever.invoke(queryy) ## get the most relevant docs ( only 3 docs)

        # Step 2: Format documents into context string
        context=format_docs(docs) ## put all the docs in one string to be a context
        
        # Step 3: Format the prompt with context and question and Invoke LLM with the formatted messages
        response=llm.invoke([HumanMessage(content=prompt_template.format(context=context,question=query))])
        return response.content




    if __name__ == "__main__":
        print("Retrieving ... ")
    # Query
        query = "what is Pinecone in machine learning?"

    #=========================================================
    # Option 0: Raw invocation without RAG
    # ========================================================
        print("\n" + "=" * 70)
        print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
        print("=" * 70)
        result_raw = llm.invoke([HumanMessage(content=query)])
        print("\nAnswer:")
        print(result_raw.content)
  
    #=========================================================
    # Option 1: Use RAG without LCEL ( WITHOUT langchain )
    # ========================================================
        print("\n" + "=" * 70)
        print("IMPLEMENTATION 1: RAG without LCEL (Manual)")
        print("=" * 70)
        result_rag_no_lcel = retrieval_chain_without_lcel(query)
        print("\nAnswer:")
        print(result_rag_no_lcel)

        

