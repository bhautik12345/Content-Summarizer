import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader,YoutubeLoader
# from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title='🦜 LangChain Summarizer',
    page_icon='🧠',
    layout='centered'
)

# Title and subtitle
st.title('🦜 LangChain: Summarize Text from YouTube or Website')
st.subheader('🔗 Enter a URL to get a concise summary and a quiz!')

groq_api_key = os.getenv('GROQ_API_KEY')

# Sidebar for settings
# with st.sidebar:
#     st.header('⚙️ Settings')
#     groq_api_key = st.text_input('🔐 Groq API Key', type='password')

# URL input
generic_url = st.text_input('🔗 Paste YouTube or Website URL here')

# Map Prompt Template
chunk_prompt = '''
Please summarize the following speech clearly and concisely:
---
Speech: "{text}"
---
Summary:
'''
map_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=chunk_prompt.strip()
)

# Final Prompt Template
final_prompt = '''
Provide a final structured summary of the entire content with the following:
1. Begin with a short introduction.
2. Provide a numbered list of key points from the content.
3. Generate at least 5 quiz questions with their answers based on the content.
---
Speech: "{text}"
'''
final_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=final_prompt.strip()
)

# Button click action
if st.button('🚀 Summarize Content'):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error('❌ Please enter both the Groq API key and a valid URL.')
    elif not validators.url(generic_url):
        st.error('⚠️ Please enter a valid URL (YouTube or any website).')
    else:
        try:
            with st.spinner('⏳ Processing the content...'):
                if 'youtube.com' in generic_url or 'youtu.be' in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        header={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                docs = loader.load()
                final_doc = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100
                ).split_documents(docs)

                llm = ChatGroq(
                    model='meta-llama/llama-4-scout-17b-16e-instruct',
                    api_key=groq_api_key
                )

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt_template,
                    combine_prompt=final_prompt_template,
                    verbose=True
                )

                output_summary = chain.run(final_doc)
                st.success('✅ Summary Generated Successfully!')
                st.markdown(f'### 📄 Summary & Quiz:\n\n{output_summary}')
        except Exception as e:
            st.error(f"🚨 An error occurred:\n\n{e}")

