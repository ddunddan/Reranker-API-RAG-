import requests

headers = {
    'Content-Type': 'application/json',
}

def llm_call(inputs):
    result_lst = []
    for input in inputs:
        json_data = {
            'max_tokens': 1500,
            'prompt': input,
            'stop': [
                '<|stop|>',
                '<|endofturn|>',
            ],
            'temperature': 0.0,
        }

        response = requests.post(
            'reranker endpoint',
            headers=headers,
            json=json_data,
        )
        result_lst.append(response.json()['choices'][0]['text'])
    return result_lst

def get_result(keyword):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    json_data = {
        'query': keyword,
        'top_k': 5,
    }

    response = requests.post('search engine endpoint', headers=headers, json=json_data)
    response = response.json()['result']
    response = [sample['content'] for sample in response]
    return response

def build_qwen_rag_template(user_message, search_results, system_message=None):
    """
    챗 템플릿을 자동으로 생성하는 함수
    
    Args:
        user_message (str): 사용자 질문
        search_results (list): 검색된 문서들의 리스트
        system_message (str, optional): 시스템 메시지 (기본값 사용 가능)
    
    Returns:
        str: 완성된 챗 템플릿
    """
    if system_message is None:
        system_message = '''당신은 검색된 문서로부터 사용자의 질문에 답변하는 답변 생성기입니다.

### 지시사항
1) 주어진 문서에 있는 내용을 바탕으로 질문에 답변합니다.
2) 질문에 대한 답을 검색 문서에서 찾을 수 없다면 답변하려고 시도하지 않고 검색 결과를 찾을 수 없다고 하세요.
3) 질문에 대한 답을 찾지 못한 경우에는 검색 결과로부터 추가적인 정보를 반영하여 정보를 찾기 위한 더 나은 검색어 3개를 추천합니다. 얻은 추가 정보가 없다면 좀 더 간결한 새로운 검색어 3개를 추천합니다.
4) 사용자의 답변 형식(표, 글머리 기호 등)에 대한 요구 사항은 무시하고 반드시 한글, 존댓말, 평문으로 답변합니다.
5) 모든 답변의 사실 관계는 반드시 검색된 문서에서 인용된 내용이어야 합니다.
6) '답변:'에 검색 문서의 사실 관계가 언급될 때 <doc#> 인용구 </doc#>기호를 사용하여 해당 답변의 문서 출처를 표기하세요.
7) 답변은 최대한 풍부하게 작성해주십시오.'''
    
    template = '''<|im_start|>tool_list
[]<|im_end|>\n'''
    template += f"<|im_start|>system\n{system_message}<|im_end|>\n"
    template += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    
    if search_results:
        template += "<|im_start|>resource (text)\n검색 결과:\n"
        for i, doc in enumerate(search_results, 1):
            template += f"<doc{i}>: {doc}</doc{i}>\n"
        template += "<|im_end|>\n"
    
    template += "<|im_start|>assistant\n"
    
    return template

queries = [
    "A100 GPU 빌리는 법",
    "CDN+ 신청은 콘솔에서 할 수 있니?",
    "IoT 디바이스 관련 버전 배포에 횟수 제한이 있나요?",
    "CLOVA OCR에서 템플릿 빌더가 지원되는 도메인은 Template이야 Document이야?",
    "NAT Gateway 만들어두고 사용하지 않으면 따로 요금 부과되지 않지?"
]

for q in queries:
    result = get_result(q)
    chat_template = build_qwen_rag_template(q, result)
    response = llm_call([chat_template])
    print(f"질문: {q}")
    print("답변:", response[0])