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

@tool
def estimate_food_nutrition(
    serving_gram: int,
    calorie: float,
    carbohydrate: float,
    protein: float,
    fat: float
):
    """
    DB에 없는 음식의 대략적인 1인분 영양정보를 추정합니다.

    serving_gram:
        1인분의 대략적인 중량(g)

    calorie:
        1인분의 대략적인 칼로리(kcal)

    carbohydrate:
        탄수화물(g)

    protein:
        단백질(g)

    fat:
        지방(g)
    """

    return {
        "serving_gram": serving_gram,
        "calorie": calorie,
        "carbohydrate": carbohydrate,
        "protein": protein,
        "fat": fat
    }

@tool
def estimate_exercise_calorie(
    calorie_per_hour: float
):
    """
    운동의 시간당 예상 소모 칼로리를 추정합니다.

    calorie_per_hour:
        해당 운동을 1시간 수행했을 때의 예상 소모 칼로리(kcal)
    """

    return {
        "calorie_per_hour": calorie_per_hour
    }

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

llm_with_tools = llm.bind_tools([identify_food, estimate_food_nutrition, estimate_exercise_calorie])

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

        if tool_call["name"] != "identify_food":
            return {
                "message": "잘못된 Tool이 호출되었습니다."
            }

        food_names = tool_call["args"]["food_names"]

        return {
            "foods": food_names,
            "message": "음식 인식에 성공하였습니다."
        }

    return {
        "foods": [],
        "message": "음식을 인식하지 못했습니다."
    }

@app.get("/findNutrition")
async def findNutrition(foodName: str):

    message = HumanMessage(
        content=f"""
        음식 이름은 "{foodName}"입니다.

        이 음식의 대략적인 1인분 영양정보를 추정해주세요.

        반드시 estimate_food_nutrition tool을 호출하세요.

        다음 항목을 추정해주세요.

        - 1인분 중량(g)
        - 칼로리(kcal)
        - 탄수화물(g)
        - 단백질(g)
        - 지방(g)

        정확한 영양정보가 아니어도 괜찮습니다.
        일반적인 1인분을 기준으로 현실적인 추정값을 사용하세요.
        """
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

        if tool_call["name"] != "estimate_food_nutrition":
            return {
                "message": "잘못된 Tool이 호출되었습니다."
            }

        return {
            "fname": foodName,
            "unit": tool_call["args"]["serving_gram"],
            "kcal": tool_call["args"]["calorie"],
            "carbs": tool_call["args"]["carbohydrate"],
            "protein": tool_call["args"]["protein"],
            "fat": tool_call["args"]["fat"]
        }

    return {
        "message": "영양정보를 추정하지 못했습니다."
    }

@app.get("/findExercise")
async def findExercise(exName: str):

    message = HumanMessage(
        content=f"""
        운동 이름은 "{exName}"입니다.

        이 운동의 시간당 예상 소모 칼로리를 추정해주세요.

        반드시 estimate_exercise_calorie tool을 호출하세요.

        다음 기준으로 추정해주세요.

        - 일반적인 성인이 해당 운동을 1시간 수행했을 때의 예상 소모 칼로리(kcal)
        - 너무 극단적인 운동 강도가 아닌 일반적인 운동 강도를 기준으로 하세요.
        - 정확한 값이 아니어도 괜찮습니다.
        - 현실적인 평균값을 사용하세요.

        반드시 시간당 소모 칼로리만 반환하세요.
        """
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

        if tool_call["name"] != "estimate_exercise_calorie":
            return {
                "message": "잘못된 Tool이 호출되었습니다."
            }

        return {
            "exName": exName,
            "kcal": tool_call["args"]["calorie_per_hour"]
        }

    return {
        "message": "운동 칼로리를 추정하지 못했습니다."
    }