import os
import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import time

import base64

# Initialize ChatOpenAI and OpenAIEmbeddings
llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo', api_key=os.environ['OPENAI_API_KEY'])

# Initialize Qdrant client and collection
client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_LOCALHOST"),
    api_key=os.getenv("QDRANT_LOCAL_API_KEY"),
    prefer_grpc=False,
)
collection_name = os.getenv("QDRANT_LOCAL_COLLECTION")



# Initialize Qdrant vector store
qdrant = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=OpenAIEmbeddings(),
    metadata_payload_key="payload",
)

# Initialize retriever and QA system
retriever = qdrant.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


import base64

LOGO_IMAGE = "Logo.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
        
    }
    .logo-text {
        font-weight:700 ;
        font-size:50px ;
        color: #f0000 ;
        
        
    }
    .logo-img {
        float:right;
        border-radius:25px;
        margin:15px ;
        width: 50px; 
        height: 50px;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">Infosys Chatbot</p>
    </div>
    """,
    unsafe_allow_html=True
)






if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    response = qa.run(question)
    # response = qa.invoke(question)
    

    
    msg = response  
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
