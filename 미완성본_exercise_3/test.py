import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Define data structure
class Hotel(BaseModel):
    introduction: str = Field(description="A brief introduction to some recommended hotels.")
    hotel_id: list = Field(description="Korean names of recommended hotels")

# Constants
FILE_PATH = "/root/LLM_Bootcamp/exercise_3/data/cleaned_top_200_rows (1).txt"
PERSIST_DIRECTORY = "/root/LLM_Bootcamp/exercise_3/data/chroma_db"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 20

def load_document(file_path):
    with open(file_path, encoding='utf-8') as f:
        return Document(page_content=f.read(), metadata={"source": "정제된 200개 호텔"})

def split_document(document):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", ","],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents([document])

def create_vector_store(documents, persist_directory):
    embedding_model = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

def init_retriever(file_path, persist_directory):
    embedding_model = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        document = load_document(file_path)
        split_docs = split_document(document)
        vector_store = create_vector_store(split_docs, persist_directory)
    
    return vector_store.as_retriever(search_type="similarity")

def create_azure_model():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )

def create_prompts():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "Answer for the question in Korean."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_system_prompt_str = """
    As an experienced hotel reviewer, carefully evaluate the provided hotel descriptions and information.
    Based on your analysis, recommend the top 3 hotels, providing detailed reasons for your choices, such as amenities, location, customer reviews, and overall value. 
    Ensure each recommendation is clearly justified. If there are no hotel names, respond with None.
    Answer for the question in Korean using the following format.\n{format_instructions}
    
    Hotel information: 
    {context} """.strip()

    qa_prompt_template = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt_str),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return contextualize_q_prompt, qa_prompt_template

global_memory = ConversationBufferMemory(chat_memory=InMemoryChatMessageHistory(), return_messages=True)


def init_chain(retriever, memory):
    azure_model = create_azure_model()
    contextualize_q_prompt, qa_prompt_template = create_prompts()
    output_parser = JsonOutputParser(pydantic_object=Hotel)

    history_aware_retriever = create_history_aware_retriever(azure_model, retriever, contextualize_q_prompt)
    qa_prompt_template = qa_prompt_template.partial(format_instructions=output_parser.get_format_instructions())
    question_answer_chain = create_stuff_documents_chain(azure_model, qa_prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    load_context_runnable = RunnablePassthrough().assign(chat_history=RunnableLambda(lambda x: memory.chat_memory.messages))

    def save_context(chain_output):
        memory.chat_memory.add_user_message(chain_output["input"])
        memory.chat_memory.add_ai_message(chain_output["answer"])
        return chain_output["answer"]

    save_context_runnable = RunnableLambda(save_context)
    return load_context_runnable | rag_chain | save_context_runnable

def get_json(query):
    global global_memory
    retriever = init_retriever(FILE_PATH, PERSIST_DIRECTORY)
    rag_chain = init_chain(retriever, global_memory)
    return rag_chain.invoke({"input": query})

if __name__ == "__main__":
    print(get_json("서울의 호텔을 추천해."))
    print(get_json("주변에 놀 곳이 있으면 좋겠어."))