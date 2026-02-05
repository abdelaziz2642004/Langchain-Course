from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# import os
load_dotenv()
information = """
Anthony Edward Stark is a fictional character primarily portrayed by Robert Downey Jr. in the Marvel Cinematic Universe (MCU) media franchise—based on the Marvel Comics character of the same name—commonly known by his alias, Iron Man. Stark is initially depicted as an industrialist, genius inventor, and former playboy who is CEO of Stark Industries. Initially the chief weapons manufacturer for the U.S. military, he has a change of heart and redirects his technical knowledge into creating mechanized suits of armor, which he uses to defend Earth.

Stark becomes a founding member and eventual leader of the Avengers. Following his failed Ultron Program, the internal conflict within the Avengers due to the Sokovia Accords, and Thanos successfully erasing half of all life in the Blip, Stark retires, marries Pepper Potts, and they have a daughter named Morgan. However, Stark rejoins the Avengers on a final mission to undo Thanos' actions. He engineers a time travel device, and the Avengers successfully restore trillions of lives across the universe before Stark ultimately sacrifices his life to defeat Thanos and his army. Stark chooses Peter Parker as a successor.

Stark is one of the central figures of the MCU, having appeared in nine films as of 2024. The character and Downey's performance have been credited with helping to cement the MCU as a multi-billion-dollar franchise, with Stark's evolution often considered the defining arc of the series. Alternate versions of Stark from within the MCU multiverse appears in various Disney+ animated series, voiced by Mick Wingert.
"""
summary_template = """
given the information about a person, summarize it in a few words and
secondly create two interesting two interesting facts about the person.
information: {information}
summary: 
"""

def main():
    print("Hello from langchain-course-1!")
    # print(os.getenv("OPENAI_API_KEY"))

# llm = ChatOpenAI(model_name="gpt-5", temperature=0)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
prompt = PromptTemplate(template=summary_template, input_variables=["information"])

# the runnable Object
chain = prompt | llm # this is a chain :D ( LCEL language )
response = chain.invoke({"information": information})
print(response)

if __name__ == "__main__":
    main()
