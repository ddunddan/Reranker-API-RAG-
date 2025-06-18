import requests

HEADERS = {
    'Content-Type': 'application/json',
}

def llm_call(prompt: str) -> str:
    """LLM(리랭커) API에 프롬프트를 전달해 답변을 받는다."""
    json_data = {
        'max_tokens': 1500,
        'prompt': prompt,
        'stop': ['<|stop|>', '<|endofturn|>'],
        'temperature': 0.0,
    }
    response = requests.post(
        'reranker endpoint',
        headers=HEADERS,
        json=json_data,
    )
    return response.json()['choices'][0]['text']

def search_documents(query: str, top_k: int = 5) -> list:
    """검색 엔진 API를 통해 관련 문서 리스트를 받아온다."""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    json_data = {
        'query': query,
        'top_k': top_k,
    }
    response = requests.post('search engine endpoint', headers=headers, json=json_data)
    return [sample['content'] for sample in response.json()['result']]

def build_rag_prompt(user_message: str, search_results: list, system_message: str = None) -> str:
    """RAG용 LLM 프롬프트 템플릿을 생성한다."""
    if system_message is None:
        system_message = (
            '당신은 검색된 문서로부터 사용자의 질문에 답변하는 답변 생성기입니다.\n'
            '1) 주어진 문서에 있는 내용을 바탕으로 질문에 답변합니다.\n'
            '2) 답을 찾지 못하면 "검색 결과를 찾을 수 없습니다"라고 답하세요.\n'
            '3) 답을 못 찾으면 더 나은 검색어 3개를 추천하세요.\n'
            '4) 반드시 한글, 존댓말, 평문으로 답변하세요.\n'
            '5) 모든 답변은 검색 문서에서 인용된 내용이어야 합니다.\n'
            '6) 인용 시 <doc#>...</doc#> 표기법을 사용하세요.\n'
            '7) 답변은 최대한 풍부하게 작성하세요.'
        )
    template = '<|im_start|>tool_list\n[]<|im_end|>\n'
    template += f"<|im_start|>system\n{system_message}<|im_end|>\n"
    template += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    if search_results:
        template += "<|im_start|>resource (text)\n검색 결과:\n"
        for i, doc in enumerate(search_results, 1):
            template += f"<doc{i}>: {doc}</doc{i}>\n"
        template += "<|im_end|>\n"
    template += "<|im_start|>assistant\n"
    return template

def run_rag_pipeline(queries: list):
    """전체 RAG 파이프라인을 실행한다."""
    for q in queries:
        docs = search_documents(q)
        prompt = build_rag_prompt(q, docs)
        answer = llm_call(prompt)
        print(f"질문: {q}")
        print("답변:", answer)
        print("-" * 40)

if __name__ == "__main__":
    # LLM 개념 관련 예시 쿼리
    example_queries = [
        "LLM과 전통적인 언어모델의 차이점은?",
        "RAG 시스템에서 리랭커의 역할은 무엇인가요?",
        "LLM이 환각(hallucination) 현상을 보이는 이유는?",
        "Prompt Engineering이란 무엇인가요?",
        "파인튜닝과 프롬프트 튜닝의 차이점은?"
    ]
    run_rag_pipeline(example_queries)