from question_answer_chain import init_retriver, init_chain, ask_something
from dotenv import load_dotenv


load_dotenv()  # .env 파일에서 환경 변수 로드

filepath = "/root/LLM_Bootcamp/exercise_3/data/cleaned_top_200_rows (1).txt"

# 리트리버 초기화
retriever = init_retriver(filepath)
# 체인 초기화
rag_chain  = init_chain(retriever)

# 사용자 입력 목록
human_inputs = [
    "서울에서 지낼만 한 호텔 추천해줘.",
    "주변에 놀 곳이 많으면 좋을 겠어.",
]

# 각 사용자 입력에 대해 질문
for input in human_inputs:
    ask_something(rag_chain, input)