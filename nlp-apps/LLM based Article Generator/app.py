import os
from apikey import OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# App framework
st.title('Wikipedia AI Assistant')
prompt_topic = st.text_input("Enter a topic below...")

# Prompt Templates
title_template_text = '''
You are and AI Assistant that generates title for an Article

Write an article headline about {topic}.
'''
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = title_template_text 
)

article_template_text = '''
You are an AI Assistant that generates an short Article on provided title

Write an short 100 words article on {title} 
'''
article_template = PromptTemplate(
    input_variables = ['title'],
    template = article_template_text 
)

# LLMs
llm = OpenAI(temperature=0.9)

# lLM chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True, output_key='article')
# simple_sequential_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)
sequential_chain = SequentialChain(chains=[title_chain, article_chain], input_variables=['topic'], output_variables=['title','article'], verbose=True)


# Passing prompt to llm and fetching response
if prompt_topic:
    # Fetching response of the final response from the chain that lies at the end. Useful it prototyping, as it doesn't require labelled outputs
    # response = title_chain.run(topic=prompt_topic)
    # response = simple_sequential_chain.run(prompt_topic)
    # st.write(response)

    # Fetching multiple responses at once. This requires labelled output for each llm chains
    response = sequential_chain({'topic':prompt_topic})
    st.write(response['title'])
    st.write(response['article'])
else:
    st.write("Please enter desired topic")