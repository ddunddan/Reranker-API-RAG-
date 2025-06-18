import streamlit as st
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import base64
import json
import http.client
import re

# 약어 처리 코드
def replace_keywords(text, keyword_dict):
    """
    Replace keywords in the given text with their corresponding values in the dictionary,
    considering only keywords that appear as independent words.

    Args:
        text (str): The input text to process.
        keyword_dict (dict): A dictionary where keys are the keywords to search for,
                             and values are the replacements.

    Returns:
        str: The text with keywords replaced.
    """
    # Create a regex pattern to match the keywords as independent words
    pattern = r"(?<![a-zA-Z0-9])(" + "|".join(map(re.escape, keyword_dict.keys())) + r")(?![a-zA-Z0-9])"


    # Replacement function using the dictionary
    def replacer(match):
        return keyword_dict[match.group(0)]

    # Substitute keywords in the text
    return re.sub(pattern, replacer, text)

# Example usage
services = {
    "CLA": "Cloud Log Analytics",
    "SENS": "Simple & Easy Notification Service",
    "NKS": "Ncloud Kubernetes Service",
    "LB": "Load Balancer",
    "CDB": "Cloud DB for",
    "CAT": "Cloud Activity Tracer",
    "RM": "Resource Manager",
    "GTM": "Global Traffic Manager",
    "ASG": "Auto Scaling Group",
    "LC": "Launch Configuration",
    "COM": "Cloud Outbound Mailer",
    "VGW": "Virtual Private Gateway",
    "SM": "Security Monitoring",
    "CM": "Certificate Manager",
    "CSW": "Cloud Security Watcher",
    "NCP": "네이버 클라우드 플랫폼",
    "CF": "Cloud Function",
    "BM": "Bare Metal Server",
    "DMS": "Database Migration Service",
    "KMS": "Key Management Service",
    "GCDN": "Global CDN",
    "WMS": "Web Service Monitoring System",
    "NTM": "Network Traffic Monitoring",
    "ELSA": "Effective Log Search & Analytics",
    "VPE": "Video Player Enhancement"
}


# 임베딩
app = FastAPI()

class ApiToolExecutor:
    def __init__(self, host, client_id, client_secret, access_token=None):
        self._host = host
        # client_id and client_secret are used to issue access_token.
        # You should not share this with others.
        self._client_id = client_id
        self._client_secret = client_secret
        # Base64Encode(client_id:client_secret)
        self._encoded_secret = base64.b64encode('{}:{}'.format(self._client_id, self._client_secret).encode('utf-8')).decode('utf-8')
        self._access_token = access_token

    def _refresh_access_token(self):
        headers = {
            'Authorization': 'Basic {}'.format(self._encoded_secret)
        }

        conn = http.client.HTTPSConnection(self._host)
        # If existingToken is true, it returns a token that has the longest expiry time among existing tokens.
        conn.request('GET', '/v1/auth/token?existingToken=true', headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()

        token_info = json.loads(body)
        self._access_token = token_info['result']['accessToken']

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': 'Bearer {}'.format(self._access_token)
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(completion_request)
        if res['status']['code'] == '40103':
            # Check whether the token has expired and reissue the token.
            self._access_token = None
            return self.execute(completion_request)
        elif res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            return 'Error'

def get_embedding(input_text):
    completion_executor = ApiToolExecutor(
        host='api-hyperclova.navercorp.com',
        client_id='047c28cc751e4779a0778451cb8c5ccd',
        client_secret='54780b564ef1d244838d8fe97a1d1a03e1d64af113cf9a82a7c5916ddd0b9b99'
    )

    #json_string = '{{"text":"{0}"}}'.format(input_text.replace('"', '\\"'))
    # request_data = json.loads(json_string, strict=False)
    
    # json.dumps를 사용하여 파이썬 객체를 JSON 문자열로 변환
    json_string = json.dumps({"text": input_text})

    # JSON 문자열을 파이썬 딕셔너리로 변환
    request_data = json.loads(json_string)

    response_data = completion_executor.execute(request_data)
    return response_data


@st.cache(allow_output_mutation=True)
def start():
    file_path = '/Users/user/Desktop/rag/ncloud_doc_and_faq_1209.csv'
    if os.path.isfile(file_path):
        print(f"{file_path} 파일이 존재합니다.")
        df = pd.read_csv(file_path)
        df['ID'] = df.index
        # df['embedding'] = df['embedding'].apply(ast.literal_eval)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        print('임베딩 파일을 찾을 수 없습니다.')
    return df

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query, top_k):
    embedding = get_embedding(replace_keywords(query, services))
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(embedding)))
    top_k_doc = df.sort_values("similarity", ascending=False).head(top_k)
    return top_k_doc

def return_results(df, query, top_k):
    result_lst = []
    result = return_answer_candidate(df, query, top_k)
    
    for i in range(0, top_k):
        result_lst.append({"id": str(result.iloc[i]['ID']), "content":str(result.iloc[i]['text'])})
    return result_lst

class SearchInput(BaseModel):
    query: str
    top_k: int

class SearchResult(BaseModel):
    id: str
    content: str

class SearchOutput(BaseModel):
    result: List[SearchResult]

df = start()

@app.post("/search/", response_model=SearchOutput)
async def search(input: SearchInput) -> SearchOutput:
    # 여기에서 입력받은 문자열 `input.text`를 사용하여 검색 로직을 구현합니다.
    # 예시를 위해 임시 데이터를 생성합니다.
    results = return_results(df, input.query, input.top_k)
    print(results)
    return SearchOutput(result=results)