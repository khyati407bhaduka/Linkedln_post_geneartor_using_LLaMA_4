from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import os


llm = ChatGroq(groq_api_key = os.getenv("GROQ_API_KEY"), model ="meta-llama/llama-4-maverick-17b-128e-instruct")


if __name__ == '__main__':
    reponse = llm.invoke("What are the two main ingredients of samosa")
    print (reponse.content)