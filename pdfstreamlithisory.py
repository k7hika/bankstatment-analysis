#pdf summary,extraction,answer question with  history but explicitly saving history
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
import os
import time
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
load_dotenv()
def loader(file):  #loading and split the pdf
    try:
        docume=PyPDFLoader(file)
        documentpdf=docume.load_and_split()
        return documentpdf
    except Exception as e:
        st.error(f"pdf loading issue{e}")
def summary(file): #function for generating summary
    try:
        llm=ChatOpenAI(api_key=os.getenv("API_KEY"),model="gpt-3.5-turbo",temperature=0)
        summarise=load_summarize_chain(llm=llm,chain_type='map_reduce')
        final_summary=summarise.invoke(file)
        return final_summary["output_text"]
    except Exception as e:
        st.error(f"Error in summary generation{e}")

def extractdoc(file):
    text=""
    for page in file:
        text+=page.page_content+"\n"
    return text
def vectorconversion(text):
    try:
        api_key=os.getenv("API_KEY")
        if not api_key:
            st.error("api key is not found, pls check your environment variable")
            return None
        embe=OpenAIEmbeddings(api_key=api_key)
        vecto=FAISS.from_texts(texts=[text],embedding=embe)
        return vecto
    except Exception as e:
        st.error(f"error in creating vector{e}")
def answe_question(vector_creation,question):
    llm=ChatOpenAI(api_key=os.getenv("API_KEY"),model="gpt-3.5-turbo",temperature=0.1)
    retrive=vector_creation.as_retriever()
    relevant_data=retrive.get_relevant_documents(question)
    context= "".join([doc.page_content for doc in relevant_data])
    histostring=""
    if st.session_state['chat_history'] is not None:
        histostring="".join([f'user_question:{hist['Question']}\nBot_answer:{hist['Answer']}' for hist in  st.session_state['chat_history']])
    temp=("You are a highly skilled financial analyst specializing in bank statement analysis.Your task is to answer user questions accurately, based strictly on the given bank statement context.\n"
          "Focus on providing concise, accurate, and professional answers. Avoid assumptions beyond the data provided.\n\n"
          "context:{context}\n\nchat_history:{history}\n\nquestion:{user_question}\nAnswer")
    prompt_template=PromptTemplate(input_variables=['context','history','user_question'],template=temp)
    chai=LLMChain(llm=llm,prompt=prompt_template)
    response=chai.run({'context':context,'history':histostring,'user_question':question})
    return response
def submit():   #helping input text box back to empty
    st.session_state['newkey']=st.session_state['query']
    st.session_state['query']=''
def main():
    st.set_page_config(page_title="Bank Statement analysis support",page_icon="📄",layout="wide")
    st.title("Bank Statement analysis")
    if 'upload_file' not in st.session_state:
        st.session_state['upload_file']=None
        st.session_state['filelo']=False
    if "extract" not in st.session_state:
        st.session_state["extract"]=None
    if "summar" not in st.session_state:
        st.session_state["extract"]=None
    if "vector_store" not in st.session_state:
        st.session_state['vector_store']=None
    if "document" not in st.session_state:
        st.session_state['document']=None
    if 'newkey' not in st.session_state: # for resetting input box to original once answer shown
        st.session_state['newkey']=''
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history']=[]
    upload_one=st.file_uploader("upload your pdf file",type="pdf")
    if upload_one is not None and st.session_state['upload_file']!=upload_one:
        st.session_state['upload_file']=upload_one
        with st.spinner("pdf uploading"):
            time.sleep(3)
        st.success("pdf uploaded successfully")
    if upload_one :
        try:
            st.write(upload_one.name)
            os.makedirs("temp",exist_ok=True)
            tempfi=os.path.join("temp",upload_one.name)
            with open(tempfi,"wb") as fi:
                fi.write(upload_one.getbuffer())
            st.session_state['document']=loader(tempfi)
            st.session_state["extract"]=extractdoc(st.session_state['document'])
            if st.session_state['vector_store'] is None:
                st.session_state['vector_store'] = vectorconversion(st.session_state['extract'])
        except Exception as e:
            st.error(f"pdf is not uploaded correctly{e}")

    col1,col2=st.columns(2)
    with col1:
        if st.button("summary"):
            with st.spinner("generating summary"):
                final=summary(st.session_state['document'])
                st.write(final)
    with col2:
        if st.button("extract"):
            st.write(st.session_state["extract"])
    st.text_input("enter your question",key="query",placeholder="type your question here",on_change=submit)
    question=st.session_state.get("newkey","")
    if question and st.session_state["vector_store"] is not None:
        if question and st.session_state["vector_store"] :
            answe=answe_question(st.session_state['vector_store'],question)
            if answe:
                st.session_state['chat_history'].append({"Question":question,"Answer":answe})
            else:
                st.error("answer is not generated pls try again")
        else:
            st.error("upload pdf and enter question")

    if st.button("answer the question"):
        if question and st.session_state["vector_store"]:
            with st.spinner("finding the answer"):
                st.write(question)
                st.write(answe)
    # if st.button("Answer the question"):
    #     if question and st.session_state["vector_store"]:
    #         with st.spinner("Finding the answer..."):
    #             answe = answe_question(st.session_state['vector_store'], question)
    #             if answe:
    #                 st.session_state['chat_history'].append({"Question": question, "Answer": answe})
    #                 st.write(f"**Q:** {question}")
    #                 st.write(f"**A:** {answe}")
    #             else:
    #                 st.error("Unable to generate an answer. Please try again.")
    #     else:
    #         st.error("Please upload a PDF and ask a valid question.")

if __name__=="__main__":
    main()


