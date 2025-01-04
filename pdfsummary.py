from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
load_dotenv()
def loader(filename):
    file_up=PyPDFLoader(filename)
    document=file_up.load_and_split()
    return document
def summary(document):
    llm=ChatOpenAI(api_key=os.getenv("API_KEY"),model="gpt-3.5-turbo",temperature=0)
    lang_data=load_summarize_chain(llm=llm,chain_type="map_reduce")
    summarydata=lang_data.invoke(document)
    return summarydata
def main():
    filename="Startup.pdf"
    loaded=loader(filename)
    summariser=summary(loaded)
    print(summariser)
if __name__=="__main__":
    main()