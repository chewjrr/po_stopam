import requests
from pprint import pprint

url = "https://api.intelligence.io.solutions/api/v1/models"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6ImZiMTIyMDhjLTM1OWUtNGEzYi05MmE2LTExMzM1YWM5YjJmNCIsImV4cCI6NDg5NjU5MjM2OX0.Bo6sddCvnPaHljwD87AmZWdNBAOiag2HNt1pJdEgCVPFQgkjLwZpy-HhfFai5D-cvxFTUpEowxguq8cun5oPYw",
}

response = requests.get(url, headers=headers)
data = response.json()

for i in range(len(data['data'])):
    name = data['data'][i]['id']
    print(name)

#Модели нейросетей

# Qwen/QwQ-32B
# meta-llama/Llama-3.2-90B-Vision-Instruct
# deepseek-ai/DeepSeek-R1
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# meta-llama/Llama-3.3-70B-Instruct
# Qwen/Qwen2-VL-7B-Instruct
# databricks/dbrx-instruct
# mistralai/Ministral-8B-Instruct-2410
# netease-youdao/Confucius-o1-14B
# nvidia/AceMath-7B-Instruct
# neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic
# mistralai/Mistral-Large-Instruct-2411
# microsoft/phi-4
# SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B
# watt-ai/watt-tool-70B
# bespokelabs/Bespoke-Stratos-32B
# NovaSky-AI/Sky-T1-32B-Preview
# tiiuae/Falcon3-10B-Instruct
# CohereForAI/c4ai-command-r-plus-08-2024
# THUDM/glm-4-9b-chat
# Qwen/Qwen2.5-Coder-32B-Instruct
# CohereForAI/aya-expanse-32b
# jinaai/ReaderLM-v2
# openbmb/MiniCPM3-4B
# Qwen/Qwen2.5-1.5B-Instruct
# ozone-ai/0x-lite
# microsoft/Phi-3.5-mini-instruct
# ibm-granite/granite-3.1-8b-instruct