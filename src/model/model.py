from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")
llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key, 
    openai_api_base=base_url,
    temperature=0.7,
    max_tokens=300,
    default_headers={
        'Ocp-Apim-Subscription-Key': api_key
    }
)

response = llm.invoke("Witaj!")
print(response.json()['choices'][0]['message']['content'])