from dotenv import load_dotenv
load_dotenv()


from langchain_core.documents import Document
def init_retriver(file_path):
    # 데이터 로드 (도구)
    with open(file_path, encoding='utf-8') as f:
            hotels_txt = f.read()

    # 데이터 객체화 (객체)
    hotel_document = Document(
        page_content=hotels_txt,
        metadata={"source": "정제된 200개 호텔"}
    )


    from langchain_text_splitters import RecursiveCharacterTextSplitter

    #RecursiveCharacterTextSplitter, chunk로 쪼개는 (도구)
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n",".",","],
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )

    # 쪼개진 문서 (객체)
    recursive_splitted_document = recursive_text_splitter.split_documents([hotel_document])


    from langchain_openai import AzureOpenAIEmbeddings
    #Embedding, 문서를 Vector로 변환해줄 모델 설정 (도구)
    embedding_model=AzureOpenAIEmbeddings(
        model="text-embedding-3-large"
    )


    from langchain_chroma import Chroma
    # Chroma 객체 생성 
    chroma = Chroma("vector_store")
    # 쪼개진 문서와 embedding 모델을 인자로 제시하면 vector로 변환 후 저장.
    vector_store = chroma.from_documents(
            documents=recursive_splitted_document,
            embedding=embedding_model
        )

    # 유사성 리트리버 생성
    similarity_retriever = vector_store.as_retriever(search_type="similarity")
    retriever = similarity_retriever
    return retriever