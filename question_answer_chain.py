from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
import json
import os

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from history_aware_retriever import init_retriver

# Define your desired data structure.
class Hotel(BaseModel):
    introduction: str = Field(description="A brief introduction to some recommended hotels.")
    hotel_id: list = Field(description="Korean names of recommended hotels")

output_parser = JsonOutputParser(pydantic_object=Hotel)


# 체인을 초기화하는 함수
def init_chain(retriever):
    # Azure OpenAI 모델 초기화
    azure_model = AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version = os.getenv("OPENAI_API_VERSION")
    )

    # 히스토리 기반 리트리버 생성
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "Answer for the question in Korean."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        azure_model, retriever, contextualize_q_prompt
    )

#----------------------------------------------------------------------------

    # 질문 응답 체인 생성
    qa_system_prompt_str = """
    As an experienced hotel reviewer, carefully evaluate the provided hotel descriptions and information.
    Based on your analysis, recommend the top 3 hotels, providing detailed reasons for your choices, such as amenities, location, customer reviews, and overall value. 
    Ensure each recommendation is clearly justified.
    Answer for the question in Korean using the following format.
    Do not output strings.\n{format_instructions}
    Delete the keyword json from the output.
    
    Hotel information: 
    {context} """.strip()


    qa_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt_str),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).partial(format_instructions=JsonOutputParser(pydantic_object=Hotel).get_format_instructions())

    question_answer_chain = create_stuff_documents_chain(azure_model, qa_prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 대화 내용을 저장할 메모리 초기화
    memory = ConversationBufferMemory(
            chat_memory=InMemoryChatMessageHistory(),        
            return_messages=True
        )

    # 이전 대화 내용을 로드하는 Runnable 정의
    load_context_runnable = RunnablePassthrough().assign(
        chat_history=RunnableLambda(lambda x:memory.chat_memory.messages)
    )

    # 체인 출력 저장 함수 정의
    def save_context(chain_output):
        memory.chat_memory.add_user_message(chain_output["input"])
        memory.chat_memory.add_ai_message(chain_output["answer"])
        return chain_output["answer"]

    # 체인 출력 저장 Runnable 정의
    save_context_runnable = RunnableLambda(save_context)

    # 전체 체인 구성
    rag_chain_with_history = load_context_runnable | rag_chain | save_context_runnable
    return rag_chain_with_history


# 사용자의 질문에 응답하는 함수
def ask_something(chain, query):
    print(f"User : {query}")

    # 체인을 통해 사용자 질문을 처리하고 출력
    chain_output = chain.invoke(
        {"input": query}
    )

    print(f"LLM : {chain_output}")
    return chain_output


def model(query):
    load_dotenv()  # .env 파일에서 환경 변수 로드

    filepath = "/root/LLM_Bootcamp/LangChain_Class/exercise_33/data/cleaned_top_200_rows.txt" ##### 경로에 맞게 고쳐야 될 부분.

    # 리트리버 초기화
    retriever = init_retriver(filepath)
    # 체인 초기화
    rag_chain  = init_chain(retriever)
    return json.loads(ask_something(rag_chain, query))
    

if __name__ == "__main__":
    print(type(model("서울에서 지낼만 한 호텔 추천해줘.")))
    # load_dotenv()  # .env 파일에서 환경 변수 로드

    # filepath = "/root/LLM_Bootcamp/LangChain_Class/exercise_33/data/cleaned_top_200_rows.txt" ##### 경로에 맞게 고쳐야 될 부분.

    # # 리트리버 초기화
    # retriever = init_retriver(filepath)
    # # 체인 초기화
    # rag_chain  = init_chain(retriever)

    # # 사용자 입력 목록
    # human_inputs = [
    #     "서울에서 지낼만 한 호텔 추천해줘.",
    #     "주변에 놀 곳이 많으면 좋을 겠어.",
    # ]

    # # 각 사용자 입력에 대해 질문
    # for input in human_inputs:
    #     ask_something(rag_chain, input)

# result
"""
User : 서울에서 지낼만 한 호텔 추천해줘.
LLM : ```json
{
  "introduction": "서울에서 지낼만한 호텔을 추천드립니다. 이 호텔들은 각기 다른 매력을 가지고 있으며, 편리한 위치와 다양한 편의시설을 자랑합니다.",
  "hotel_id": [
    "IBC호텔",
    "가락관광호텔",
    "명동 인근 호텔"
  ]
}
```
User : 주변에 놀 곳이 많으면 좋을 겠어.
LLM : ```json
{
  "introduction": "서울에서 지낼만한 호텔을 추천드립니다. 이 호텔들은 편리한 위치에 있으며, 주변에 다양한 관광 명소와 쇼핑 장소가 있어 즐길거리가 많습니다.",
  "hotel_id": [
    "명동 인근 호텔",
    "IBC호텔",
    "가락관광호텔"
  ]
}
```
"""
