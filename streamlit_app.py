import os
import torch
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable

from make_rag_docs import get_rag_data1
from filter_data import *

from langchain_openai import ChatOpenAI
import toml
config = toml.load('.secret/secrets.toml')

api_key = config['OPENAI_API_KEY']


# ⭐️ Embedding 설정
USE_BGE_EMBEDDING = True


# ⭐️ LangServe 모델 설정(EndPoint)
# ngrok http --domain=quick-alien-cunning.ngrok-free.app 8000
LANGSERVE_ENDPOINT = "https://quick-alien-cunning.ngrok-free.app/llm/"


# 필수 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 질문에 대해 간결하게 답하세요.
RAG_PROMPT_TEMPLATE = """당신은 던전앤파이터 게임에 관련된 질문에 답변하는 AI 입니다. 주어진 Context 내용을 참고하여 답하세요. Context에서 참고했다는 이야기는 하지마세요. 주어진 Context가 없다면 '제가 알고 있는 정보가 없습니다.'로 답변하세요.
Context: {context} 
Human: {question} 
Assistant:"""


st.set_page_config(page_title="DnF Chatbot", page_icon="📔")
st.title("📔 DnF Chatbot")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


@st.cache_resource(show_spinner='Get Embedding Model..')
def get_embed_model():

    model_name = "BAAI/bge-m3"
    # GPU Device 설정:
    # - NVidia GPU: "cuda"
    # - Mac M1, M2, M3: "mps"
    # - CPU: "cpu"
    model_kwargs = {
        # "device": "cuda"
        "device": "mps"
        # "device": "cpu"
    }
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
        
    return embeddings


def embed_file(embeddings, **kwargs):
    
    cache_dir = LocalFileStore(f"./.cache/dnf/embedding_3000_withWiki/")
    
    docs = get_rag_data1()

    torch.mps.empty_cache()
    
    bm25_retriever = BM25Retriever.from_documents(
        docs
    )
    bm25_retriever.k = 2
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    
    if kwargs:
        retriever = vectorstore.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={
                # 'score_threshold': 0.5,
                # 'score_threshold': 0.2,
                'k' : 10,
                'filter' : kwargs
                }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'score_threshold': 0.6,
                'k' : 2,
                }
        )
        
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.3, 0.7] # 키워드 검색, 문장 검색
    )
        
    
    return ensemble_retriever


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    context = ''
    for doc in docs:
        context += doc.page_content
        context += str(doc.metadata)
        context += "\n\n"
    return context[:-2]


embedding = get_embed_model()
print_history()



st.sidebar.write('검색 영역을 선택해 주세요. 보다 정확한 답변을 생성할 수 있습니다.')


search_cate1 = st.sidebar.selectbox("검색 카테고리를 선택해주세요.", options = SELECT_CATEGORY1)

if search_cate1 == '공략':
    search_cate2 = st.sidebar.selectbox("검색할 던전을 선택하세요", options=SELECT_CATEGORY2)


    SELECT_CATEGORY3 = CATEGORY3_DICT[search_cate2]
    
    search_cate3 = st.sidebar.selectbox(f"{search_cate2} 레이드를 선택하세요", options=SELECT_CATEGORY3)
    
else:
    
    SELECT_CATEGORY3 = CATEGORY3_DICT[search_cate1]
    
    search_cate3 = st.sidebar.selectbox(f"{search_cate1} 관련 궁금한 부분을 선택하세요.", options=SELECT_CATEGORY3)
    



if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        llm = RemoteRunnable(LANGSERVE_ENDPOINT)
        # llm = ChatOpenAI(model='gpt-4', openai_api_key = api_key)
        chat_container = st.empty()
        
        print('categories')
        # print(search_cate1, search_cate2, search_cate3)
        # user_input += ' '
        # if search_cate1 : user_input += search_cate1
        # if search_cate2 : user_input += search_cate2
        # if search_cate3 : user_input += search_cate3
        
        with st.spinner("Searching DnF Homepage article..."):
            if search_cate1 != '':
                if search_cate1 == '공략':
                    print('123 kwarg')
                    retriever = embed_file(embedding, 
                                       category1 = search_cate1, 
                                       category2 = search_cate2, 
                                       category3 = search_cate3,
                                       )
                else:
                    print('13 kwarg')
                    retriever = embed_file(embedding, 
                                       category1 = search_cate1, 
                                       category3 = search_cate3, 
                                       )
            else:
                print('None kwarg')
                retriever = embed_file(embedding)
            # retriever = embed_file(embedding)
        
        
        print(user_input)
        # st.write(retriever.invoke(user_input))
        
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        if len(retriever.invoke(user_input)):
            # 체인을 생성합니다.
            rag_chain = (
                {
                    # "context" : lambda x : format_docs(docs),
                    "context": retriever | format_docs,
                    # 'context' : lambda x : '',
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
                
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            with st.spinner('답변 생성 중...'):
                answer = rag_chain.stream(user_input)  # 문서에 대한 질의
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                    chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))
        else:
            NONE_PROMPT_TEMPLATE = f"""말씀해주신 질문에 대해 제가 참고할 수 있는 내용이 없습니다."""
            chat_container.markdown(NONE_PROMPT_TEMPLATE)
            add_history("ai", "".join(NONE_PROMPT_TEMPLATE))