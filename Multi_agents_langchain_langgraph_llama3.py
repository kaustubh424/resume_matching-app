import warnings
warnings.filterwarnings("ignore")
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_groq import ChatGroq
from typing import Annotated, Literal, TypedDict, Sequence
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.messages import BaseMessage
import operator
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from IPython.display import Image, display
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="llama3-70b-8192")


def load_image(image_file):
    img = Image.open(image_file)
    return img


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def rag_output(AgentState):
    
    pdf_file = "Resume.pdf"
    data = PyPDFLoader(pdf_file).load()
    
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)


    try:
        data_chunks = text_splitter.split_documents(data)
        
        vectorstore = Chroma.from_documents(documents=data_chunks, collection_name="my-rag-chroma", embedding=embeddings, persist_directory="db")
        
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory)
      
        query = "What is the LinkedIn profile link of the candidate"
        result = conversation_chain.invoke({"question": query})
        answer = result["answer"]
        

    except Exception as ex:
        print("Errors = {}".format(ex))
        

    return {"messages": [answer]}


def agent(AgentState):
    
    try:

        pdf_file = "Resume.pdf"
        data = PyPDFLoader(pdf_file).load()

        string_text = [data[i].page_content for i in range(len(data))]
        resume_data = " ".join([text for text in string_text])

        response = llm.invoke("Your task is to extract the candidate name from the resume data. Only respond with the candidate name and nothing else. Resume Data: " + resume_data)

    except Exception as ex:
        print("Errors = {}".format(ex))
        
    return {"messages": [response.content]}


def JD_agent(AgentState):
    
    try:
        with open("JD.txt", "r") as f:
            data = f.read()
            f.close()
        
        response = llm.invoke("Your task is to extract the exact job requirements from the given data. Only respond with the job requirements and nothing else. Data: " + str(data))
        result = response.content
        result = result.replace("\n", "")
        
        print("Done with job requirements extraction")
        
    except Exception as ex:
        print("Errors = {}".format(ex))

    return {"messages": [result]}
    

def web_scrap(AgentState):
    
    
    try:
        messages = AgentState['messages']
        query = messages[-1]
        output = llm.invoke("Your task is to extract the exact LinkedIn profile url from the given user query. Only respond with the url link and nothing else. User query: " + query)
        url = output.content
        loader = WebBaseLoader(url)
        scrape_data = loader.load()
        
        name = messages[1]
        print("name = {}".format(name))
        
        if str(name).lower() in str(scrape_data).lower():
            answer = "LinkedIn profile of the candidate exists"
        else:
            answer = "Can't find the LinkedIn profile"

    except Exception as ex:
        print("Errors = {}".format(ex))
        
    return {"messages": [answer]}
    

def recruit_agent(AgentState):
    
    
    try:
        pdf_file = "Resume.pdf"
        data = PyPDFLoader(pdf_file).load()
    
        string_text = [data[i].page_content for i in range(len(data))]
        resume_data = " ".join([text for text in string_text])
        
        messages = AgentState['messages']
        data = messages[-2:]
        data = str(data)   
        output = llm.invoke("Your task is to go through the resume data of the candidate and also the data which contain job requirements and if the LinkedIn profile of the candidate exists or not, then please give your judgment if the candidate is a good fit for the job. Resume Data: " + resume_data + "Data: " + data)
    except Exception as ex:
        print("Errors = {}".format(ex))

    return {"messages": [output.content]}



def main():

    st.title("App for Resume Matching for a given Role")
    html_temp = """
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;">Application for Resume matching for a given Role with Langchain, LangGraph, RAG and Llama3 model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    pdf_file = st.file_uploader("Upload Resume", type=["pdf"])
    
    if ((pdf_file is not None) and ('pdf' in str(pdf_file))):
        bytes_data = pdf_file.read()
        
        with open("Resume.pdf", 'wb') as f: 
            f.write(bytes_data)
            f.close()

    
    text_file = st.file_uploader("Upload Job Description", type=["txt"])
    
    if ((text_file is not None) and ('txt' in str(text_file))):

        text_data = str(text_file.read(), "utf-8", errors='ignore')
        text_data = text_data.replace("\n", "")
        text_data = text_data.replace("\t", "")
        text_data = text_data.replace("\s", "")
        
        text_file_path = "JD.txt"
        with open(text_file_path, "w") as f:
            f.write(text_data)
            f.close()
            
    
    if st.button("Match Resume"):

        inputs = {"messages": ["Your are a recruitment expert and your role is to match a candidate's profile with a given job description"]}
    
        workflow = StateGraph(AgentState)
        
        # Define the two nodes we will cycle between
        workflow.add_node("Resume_agent", agent)
        workflow.add_node("JD_agent", JD_agent)
        workflow.add_node("RAG_agent", rag_output)
        workflow.add_node("Web_scrap_agent", web_scrap)
        workflow.add_node("Recruiter_agent", recruit_agent)
        workflow.set_entry_point("Resume_agent")
        workflow.add_edge("Resume_agent", 'JD_agent')
        workflow.add_edge("JD_agent", "Recruiter_agent")
        workflow.add_edge("Resume_agent", 'RAG_agent')
        workflow.add_edge("RAG_agent", "Web_scrap_agent")
        workflow.add_edge("Web_scrap_agent", "Recruiter_agent")
        workflow.add_edge("Recruiter_agent", END)
        app = workflow.compile()
        img = app.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(img)
        filename = "workflow.png"        
        st.image(load_image(filename))

        outputs = app.stream(inputs)
        results = []
        for output in outputs:
            for key,value in output.items():
                
                results.append("Output from node "+str(key)+":  "+str(value['messages']) + "  ")
            
 
        st.success('Results: {}'.format(str(results)))

        
if __name__=='__main__':
    main()


