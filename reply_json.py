from question_answer_chain import init_retriver, init_chain, ask_something
from dotenv import load_dotenv


def get_json(query):
    load_dotenv()  # .env 파일에서 환경 변수 로드

    filepath = "/root/LLM_Bootcamp/exercise_3/data/cleaned_top_200_rows (1).txt"

    # 리트리버 초기화
    retriever = init_retriver(filepath)
    # 체인 초기화
    rag_chain  = init_chain(retriever)

    # 체인을 통해 사용자 질문을 처리하고 출력
    chain_output = rag_chain.invoke(
        {"input": query}
    )

    return chain_output