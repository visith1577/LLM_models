import os
from keys import OPEN_AI_API_KEY

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY

st.title("LLM - Novel writer app")
prompt = st.text_input("plug in your prompt here")

title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a title for the following topic : {topic}"
)

script_template = PromptTemplate(
    input_variables=['script'],
    template="Write a Script for the following topic : {script}"
)

script_complete_template = PromptTemplate(
    input_variables=['complete'],
    template="Complete the rest of the following convert it to comedy: {complete}"
)

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True
)
script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True
)

script_comp_chain = LLMChain(
    llm=llm,
    prompt=script_complete_template,
    verbose=True
)

sequential_chains = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)


if prompt:
    response = sequential_chains.run(prompt)
    st.write(response)
    response1 = script_comp_chain.run(response)
    st.write(response1)
