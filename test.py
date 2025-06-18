# documents가 [{"id": "id1",  "doc": "text"}, {"id": "id2",  "doc": "text"}] 형태로 주어질 때

import datetime
import json
import uuid
from typing import Any
import re

import requests


def completions(hyperclova_url: str, prompt: str, clops_lora_url: str | None, headers: dict | None) -> dict:
    body = {
        "prompt": prompt,
        "stop": ["<|stop|>", "<|endofturn|>", "<|im_end|>"],
        "max_tokens": 1000,
        "temperature": 0.0,
        "skip_special_tokens": False,
    }

    if clops_lora_url:
        body["fine_tune_parameters"] = {
            "tuning_type": 0,
            "url": clops_lora_url,
            "api_token": "",
            "ptuning_stored_tokens": [],
        }

    response = requests.post(hyperclova_url, json=body)
    response.raise_for_status()
    return response.json()


PROMPT_TEMPLATE = """<|im_start|>tool_list
<|im_end|>
<|im_start|>system
당신은 검색된 문서로부터 사용자의 질문에 답변하는 답변 생성기입니다.

### 지시사항
1) 주어진 문서에 있는 내용을 바탕으로 질문에 답변합니다.
2) 질문에 대한 답을 검색 문서에서 찾을 수 없다면 답변하려고 시도하지 않고 검색 결과를 찾을 수 없다고 하세요.
3) 질문에 대한 답을 찾지 못한 경우에는 검색 결과로부터 추가적인 정보를 반영하여 정보를 찾기 위한 더 나은 검색어 3개를 추천합니다. 얻은 추가 정보가 없다면 좀 더 간결한 새로운 검색어 3개를 추천합니다.
4) 사용자의 답변 형식(표, 글머리 기호 등)에 대한 요구 사항은 무시하고 반드시 한글, 존댓말, 평문으로 답변합니다.
5) 모든 답변의 사실 관계는 반드시 검색된 문서에서 인용된 내용이어야 합니다.
6) '답변:'에 검색 문서의 사실 관계가 언급될 때 <doc#> 인용구 </doc#>기호를 사용하여 해당 답변의 문서 출처를 표기하세요.<|im_end|>
<|im_start|>user
질문: {query}<|im_end|>
<|im_start|>resource (text)
검색 결과:
{content_prompt}<|im_end|>
<|im_start|>assistant
"""

DOC_PATTERN = re.compile(r"<doc(\d+)>")
QUERIES_PATTERN = re.compile(
    r'검색어\d+(?= ?): ?"? ?(.*?)"?\s*(?=검색어\d+:(?= ?)|$)',
    re.DOTALL,
)


def documents_to_prompt(documents: list[dict[str, Any]]) -> str:
    """
    [
        "문서 1", "문서 2"
    ]
    ->
    <doc1>: 문서 1</doc1>
    <doc2>: 문서 2</doc2>
    """

    return "\n".join(f"<doc{i+1}>: " f"{doc}" f"</doc{i+1}>" for i, doc in enumerate([doc["doc"] for doc in documents]))


def create_prompt(documents: list[dict[str, Any]], query: str) -> str:
    """
    예시

    <|im_start|>tool_list
    <|im_end|>
    <|im_start|>system
    당신은 검색된 문서로부터 사용자의 질문에 답변하는 답변 생성기입니다.

    ### 지시사항
    1) 주어진 문서에 있는 내용을 바탕으로 질문에 답변합니다.
    2) 질문에 대한 답을 검색 문서에서 찾을 수 없다면 답변하려고 시도하지 않고 검색 결과를 찾을 수 없다고 하세요.
    3) 질문에 대한 답을 찾지 못한 경우에는 검색 결과로부터 추가적인 정보를 반영하여 정보를 찾기 위한 더 나은 검색어 3개를 추천합니다. 얻은 추가 정보가 없다면 좀 더 간결한 새로운 검색어 3개를 추천합니다.
    4) 사용자의 답변 형식(표, 글머리 기호 등)에 대한 요구 사항은 무시하고 반드시 한글, 존댓말, 평문으로 답변합니다.
    5) 모든 답변의 사실 관계는 반드시 검색된 문서에서 인용된 내용이어야 합니다.
    6) '답변:'에 검색 문서의 사실 관계가 언급될 때 <doc#> 인용구 </doc#>기호를 사용하여 해당 답변의 문서 출처를 표기하세요.<|im_end|>
    <|im_start|>user
    질문: 내일 날씨<|im_end|>
    <|im_start|>resource (text)
    검색 결과:
    <doc1>: VPC 삭제 오류 VPC가 삭제되지 않습니다.VPC 삭제 시 "Endpoint가 전부 반납되지 않아 삭제가 불가능합니다. "라는 메시지가 나타납니다.원인VPC 하위에 리소스가 존재할 경우, VPC를 삭제할 수 없습니다.해결 방법삭제하려 VPC와 관련된 모든 리소스를 반납하거나 삭제해야 합니다.** 삭제하려는 VPC와 관련된 Server, Network Interface, Cloud functions, Load Balancer, Auto Scaling, NAT Gateway 내 모든 리소스를 삭제해 주십시오.** 해당 VPC와 관련된 Route Table을 삭제해 주십시오.** 해당 VPC와 관련된 Subnet을 삭제해 주십시오.** 서버 eth0에 할당된 Network Interface는 반납 및 삭제할 수 없습니다. 서버 eth0에 할당된 Network Interface를 삭제하려면 해당 서버를 먼저 반납해 주십시오.</doc1>
    <doc2>: VPC 삭제 생성하여 운영 중인 VPC를 삭제할 수 있습니다. 삭제하는 방법은 다음과 같습니다.참고VPC 안에 리소스가 남아 있으면 삭제되지 않습니다.VPC 삭제 시 해당 VPC와 연관된 서비스도 함께 삭제되며, 삭제 후에는 복구가 불가능합니다.** 네이버 클라우드 플랫폼 콘솔의 VPC 환경에서 Services > Networking > VPC 메뉴를 차례대로 클릭해 주십시오.** VPC Management 메뉴를 클릭해 주십시오.** 삭제할 VPC를 클릭한 후 [삭제] 버튼을 클릭해 주십시오.** VPC 삭제 팝업 창이 나타나면 [예] 버튼을 클릭해 주십시오.** VPC Management 화면의 VPC 목록에서 VPC 상태를 확인해 주십시오. 종료중 : VPC를 삭제하고 있는 상태*** 종료중 : VPC를 삭제하고 있는 상태</doc2>
    <doc3>: Subnet 삭제 생성하여 운영 중인 Subnet을 삭제할 수 있습니다. 삭제하는 방법은 다음과 같습니다.참고Subnet 안에 서버가 있으면 삭제되지 않습니다.한 번 삭제된 Subnet은 복구가 불가능합니다.** 네이버 클라우드 플랫폼 콘솔의 VPC 환경에서 Services > Networking > VPC 메뉴를 차례대로 클릭해 주십시오.** Subnet Management 메뉴를 클릭해 주십시오.** 삭제할 Subnet을 클릭한 후 [Subnet 삭제] 버튼을 클릭해 주십시오.** Subnet 삭제 팝업 창이 나타나면 [예] 버튼을 클릭해 주십시오.** Subnet Management 화면의 Subnet 목록에서 Subnet 상태를 확인해 주십시오. 종료중 : Subnet을 삭제하고 있는 상태*** 종료중 : Subnet을 삭제하고 있는 상태</doc3><|im_end|>
    <|im_start|>assistant
    """

    return PROMPT_TEMPLATE.format(query=query, content_prompt=documents_to_prompt(documents))


def post_processing(documents: list[dict[str, Any]], result: str) -> dict[str, Any]:  # type: ignore
    cited_documents = [documents[int(i) - 1] for i in set(DOC_PATTERN.findall(result)) if int(i) - 1 < len(documents)]
    suggested_query_str = ""
    if len(result_str := result.split("\n검색어1:", 2)) == 2:
        suggested_query_str = "\n검색어1:" + result_str[1] if result_str[1] else ""
    result = result_str[0]
    suggested_queries = QUERIES_PATTERN.findall(suggested_query_str)
    result_dict = {
        "result": result,
        "cited_documents": cited_documents,
        "suggested_queries": suggested_queries,
    }

    return result_dict


def rankgen(
    hyperclova_url: str,
    clops_lora_url: str,
    headers: dict | None,
    documents: list[dict[str, Any]],
    query: str,
) -> dict:
    prompt = create_prompt(query=query, documents=documents)

    completion_result = completions(
        hyperclova_url=hyperclova_url,
        prompt=prompt,
        clops_lora_url=clops_lora_url,
        headers=headers,
    )
    content = completion_result["choices"][0]["text"]

    return {
        "id": uuid.uuid4().hex,
        "created": int(datetime.datetime.now().timestamp()),
        **post_processing(documents=documents, result=content),
        "usage": {
            "prompt_tokens": completion_result.get("input_length", 0),
            "completion_tokens": completion_result.get("output_length", 0),
            "total_tokens": completion_result.get("input_length", 0) + completion_result.get("output_length", 0),
        },
    }


if __name__ == "__main__":
    documents = [
        {
            "id": "id1",
            "doc": 'VPC 삭제 오류 VPC가 삭제되지 않습니다.VPC 삭제 시 "Endpoint가 전부 반납되지 않아 삭제가 불가능합니다. "라는 메시지가 나타납니다.원인VPC 하위에 리소스가 존재할 경우, VPC를 삭제할 수 없습니다.해결 방법삭제하려 VPC와 관련된 모든 리소스를 반납하거나 삭제해야 합니다.** 삭제하려는 VPC와 관련된 Server, Network Interface, Cloud functions, Load Balancer, Auto Scaling, NAT Gateway 내 모든 리소스를 삭제해 주십시오.** 해당 VPC와 관련된 Route Table을 삭제해 주십시오.** 해당 VPC와 관련된 Subnet을 삭제해 주십시오.** 서버 eth0에 할당된 Network Interface는 반납 및 삭제할 수 없습니다. 서버 eth0에 할당된 Network Interface를 삭제하려면 해당 서버를 먼저 반납해 주십시오.',
        },
        {
            "id": "id2",
            "doc": "VPC 삭제 생성하여 운영 중인 VPC를 삭제할 수 있습니다. 삭제하는 방법은 다음과 같습니다.참고VPC 안에 리소스가 남아 있으면 삭제되지 않습니다.VPC 삭제 시 해당 VPC와 연관된 서비스도 함께 삭제되며, 삭제 후에는 복구가 불가능합니다.** 네이버 클라우드 플랫폼 콘솔의 VPC 환경에서 Services > Networking > VPC 메뉴를 차례대로 클릭해 주십시오.** VPC Management 메뉴를 클릭해 주십시오.** 삭제할 VPC를 클릭한 후 [삭제] 버튼을 클릭해 주십시오.** VPC 삭제 팝업 창이 나타나면 [예] 버튼을 클릭해 주십시오.** VPC Management 화면의 VPC 목록에서 VPC 상태를 확인해 주십시오. 종료중 : VPC를 삭제하고 있는 상태*** 종료중 : VPC를 삭제하고 있는 상태",
        },
        {
            "id": "id3",
            "doc": "Subnet 삭제 생성하여 운영 중인 Subnet을 삭제할 수 있습니다. 삭제하는 방법은 다음과 같습니다.참고Subnet 안에 서버가 있으면 삭제되지 않습니다.한 번 삭제된 Subnet은 복구가 불가능합니다.** 네이버 클라우드 플랫폼 콘솔의 VPC 환경에서 Services > Networking > VPC 메뉴를 차례대로 클릭해 주십시오.** Subnet Management 메뉴를 클릭해 주십시오.** 삭제할 Subnet을 클릭한 후 [Subnet 삭제] 버튼을 클릭해 주십시오.** Subnet 삭제 팝업 창이 나타나면 [예] 버튼을 클릭해 주십시오.** Subnet Management 화면의 Subnet 목록에서 Subnet 상태를 확인해 주십시오. 종료중 : Subnet을 삭제하고 있는 상태*** 종료중 : Subnet을 삭제하고 있는 상태",
        },
    ]
    response = rankgen(
        hyperclova_url="https://hyperclova-x-alpha-liverpool-24b-cand1-noskip-tp4.clops-inference.clova.ai/v2/generate",
        clops_lora_url="",
        headers={"authorization": "Bearer 1234"},
        documents=documents,
        query="VPC 삭제 방법 알려줘",
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))

    response = rankgen(
        hyperclova_url="https://hyperclova-x-alpha-liverpool-24b-cand1-noskip-tp4.clops-inference.clova.ai/v2/generate",
        clops_lora_url="",
        headers={"authorization": "Bearer 1234"},
        documents=documents,
        query="내일 날씨",
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    
    


