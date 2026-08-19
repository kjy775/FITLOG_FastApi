import base64
from typing import List

from fastapi import FastAPI, UploadFile, File
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

@tool
def identify_food(food_names: List[str]):
    """
    음식 사진에 포함된 모든 음식의 이름을 반환합니다.
    음식의 양이나 중량은 판단하지 않습니다.
    """
    return food_names

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

llm_with_tools = llm.bind_tools([identify_food])

@app.post("/findFood")
async def analyze_food(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
                이 음식 사진을 분석해주세요.

                사진에 있는 음식들을 모두 찾아주세요.

                각각의 음식 이름을 하나씩 구분해서 반환해주세요.

                음식의 양이나 중량은 판단하지 마세요.
                음식 이름만 판단해주세요.

                반드시 identify_food tool을 호출해야 합니다.

                예를 들어 사진에
                닭가슴살, 현미밥, 샐러드가 있다면

                [
                    "닭가슴살",
                    "현미밥",
                    "샐러드"
                ]

                형태로 반환해야 합니다.

                음식을 정확하게 판단할 수 없는 경우에는
                해당 음식을 목록에 포함하지 마세요.
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file.content_type};base64,{image_base64}"
                }
            }
        ]
    )

    response = llm_with_tools.invoke([message])

    print("====================================")
    print("LLM 응답")
    print(response)
    print("====================================")

    print("Tool 호출")
    print(response.tool_calls)
    print("====================================")

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        food_names = tool_call["args"]["food_names"]

        return {
            "foods": food_names,
            "message": "음식 인식에 성공하였습니다."
        }

    return {
        "foods": [],
        "message": "음식을 인식하지 못했습니다."
    }