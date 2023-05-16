import os
from keys import OPEN_AI_API_KEY

from langchain.llms import OpenAI
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY

llm = OpenAI(temperature=0.9)
loader = PyPDFLoader("annualreport.pdf")
pages = loader.load_and_split()

store = Chroma.from_documents(pages, collection_name="annualreport")

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Investment Banker')

prompt = st.text_input("Input yor prompt here")

if prompt:
    response = llm(prompt)
    st.write(response)

    with st.expander("Document Similarity Search"):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)
