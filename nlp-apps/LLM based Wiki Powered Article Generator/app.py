import os
from apikey import OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

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

Write an short 100 words article on {title} while leveraging this wikipedia research: {wikipedia_research}
'''
article_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template = article_template_text 
)

# LLMs
llm = OpenAI(temperature=0.9)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
article_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# lLM chains
# I/P: title_template O/P: title
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
# I/P: title_prompt O/P: wikipedia_research
wiki = WikipediaAPIWrapper()
# I/P: [title, wikipedia_research] O/P: article
article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True, output_key='article', memory=article_memory)


# Passing prompt to llm and fetching response
if prompt_topic:
    # 1st Chain exec: Generates title based on prompt topic provided
    title_response = title_chain.run(topic=prompt_topic)
    # 2nd Chain exec: Extract Source Wiki Pages based on Prompt (cannot mention input variable as its not defined explicitly)
    wiki_research = wiki.run(prompt_topic)
    # 3rd chain exec: Generating Article based on title_response and wiki_research
    final_response = article_chain.run(title=title_response, wikipedia_research=wiki_research)

    st.write(title_response)
    st.write(final_response)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(article_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)