import os
import pickle
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()


from langchain_core.documents import Document

def init_retriever(file_path, embed_file_path):
    # 로컬에 임베딩된 파일이 존재하면 이를 로드하고 retriever를 반환
    if os.path.exists(embed_file_path):
        with open(embed_file_path, 'rb') as f:
            vector_store_data = pickle.load(f)
        vector_store = Chroma.from_persistent(vector_store_data)
        return vector_store.as_retriever(search_type="similarity")
    
    # 로컬에 임베딩된 파일이 없으면 새로운 임베딩을 생성
    with open(file_path, encoding='utf-8') as f:
        hotels_txt = f.read()

    hotel_document = Document(
        page_content=hotels_txt,
        metadata={"source": "정제된 200개 호텔"}
    )

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", ","],
        chunk_size=400,
        chunk_overlap=20,
        length_function=len,
    )

    recursive_splitted_document = recursive_text_splitter.split_documents([hotel_document])

    embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    chroma = Chroma("vector_store")
    vector_store = chroma.from_documents(
        documents=recursive_splitted_document,
        embedding=embedding_model,
        persist_directory="/root/LLM_Bootcamp/exercise_3/data/chroma_db"
    )

    # 임베딩된 내용을 로컬 파일로 저장
    with open(embed_file_path, 'wb') as f:
        pickle.dump(vector_store, f)

    return vector_store.as_retriever(search_type="similarity")