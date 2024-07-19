from question_answer_chain import init_chain, ask_something
from history_aware_retriever import init_retriever
from dotenv import load_dotenv


def get_json(query):
    load_dotenv()  # .env 파일에서 환경 변수 로드

    # 파일 경로 설정
    file_path = "/root/LLM_Bootcamp/exercise_3/data/cleaned_top_200_rows (1).txt"
    embed_file_path = "/root/LLM_Bootcamp/exercise_3/data/chroma_db/file.pkl"

    # 리트리버 초기화
    retriever = init_retriever(file_path, embed_file_path)
    # 체인 초기화
    rag_chain  = init_chain(retriever)

    # 체인을 통해 사용자 질문을 처리하고 출력
    chain_output = rag_chain.invoke(
        {"input": query}
    )

    return chain_output


if __name__ == "__main__":
    print(get_json("지하철과 거리가 가까운 서울의 호텔을 추천해."))

