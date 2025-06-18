import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
import requests
from test import rankgen
from search import search, SearchInput, SearchResult

app = FastAPI()

class ChainInput(BaseModel):
    query: str
    top_k: int = 3
    hyperclova_url: str
    clops_lora_url: Optional[str] = None
    headers: Optional[dict] = None

class ChainOutput(BaseModel):
    id: str
    created: int
    result: str
    cited_documents: List[dict]
    suggested_queries: List[str]
    usage: dict

@app.post("/chain/", response_model=ChainOutput)
async def chain_search_and_generate(input: ChainInput) -> ChainOutput:
    # 1. 검색 수행
    search_results = await search(SearchInput(query=input.query, top_k=input.top_k))
    
    # 2. 검색 결과를 documents 형식으로 변환
    documents = [
        {
            "id": result.id,
            "doc": result.content
        }
        for result in search_results.result
    ]
    
    # 3. rankgen을 사용하여 답변 생성
    response = rankgen(
        hyperclova_url=input.hyperclova_url,
        clops_lora_url=input.clops_lora_url,
        headers=input.headers,
        documents=documents,
        query=input.query
    )
    
    return ChainOutput(**response)

def test_api():
    """
    API 테스트를 위한 함수
    서버가 실행 중일 때 이 함수를 호출하여 API를 테스트할 수 있습니다.
    """
    # 테스트용 입력 데이터
    test_input = {
        "query": "VPC 삭제 방법 알려줘",
        "top_k": 3,
        "hyperclova_url": "https://hyperclova-x-alpha-liverpool-24b-cand1-noskip-tp4.clops-inference.clova.ai/v2/generate",
        "headers": {"authorization": "Bearer YOUR_API_KEY"}
    }
    
    # API 호출
    response = requests.post(
        "http://localhost:8000/chain/",
        json=test_input
    )
    
    # 결과 출력
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # 서버가 실행된 후에 테스트를 실행하려면 다음 줄의 주석을 해제하세요
    # test_api()

"""
API 호출 예시:

1. Python requests를 사용한 호출:
```python
import requests

response = requests.post(
    "http://localhost:8000/chain/",
    json={
        "query": "VPC 삭제 방법 알려줘",
        "top_k": 3,
        "hyperclova_url": "https://hyperclova-x-alpha-liverpool-24b-cand1-noskip-tp4.clops-inference.clova.ai/v2/generate",
        "headers": {"authorization": "Bearer YOUR_API_KEY"}
    }
)
print(response.json())
```

2. curl을 사용한 호출:
```bash
curl -X POST "http://localhost:8000/chain/" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "VPC 삭제 방법 알려줘",
         "top_k": 3,
         "hyperclova_url": "https://hyperclova-x-alpha-liverpool-24b-cand1-noskip-tp4.clops-inference.clova.ai/v2/generate",
         "headers": {"authorization": "Bearer YOUR_API_KEY"}
     }'
```

3. FastAPI 자동 생성 문서:
- 서버 실행 후 브라우저에서 http://localhost:8000/docs 접속
- /chain/ 엔드포인트를 선택하여 API 테스트 가능
""" 