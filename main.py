import base64
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

import logging
import os

import httpx
import psycopg2
from pydantic import BaseModel

app = FastAPI()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrition_rag")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "fitlog"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://localhost:8070")
SPRING_TIMEOUT = float(os.getenv("SPRING_TIMEOUT", "5.0"))

TOP_K = 5
SIMILARITY_THRESHOLD = 0.3  # 이 미만이면 관련 정보 없음으로 처리

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 내부 데이터 조회 함수 (tool 내부에서 사용)
# ---------------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def search_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = embeddings.embed_query(query)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT document_id, category, subcategory, content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM nutrition_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k)
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return [
        {
            "document_id": r[0],
            "category": r[1],
            "subcategory": r[2],
            "content": r[3],
            "similarity": r[4],
        }
        for r in rows
    ]


def spring_get(path: str, params: dict | None = None):
    """
    Spring 서버 조회 공용 헬퍼.
    - 데이터가 없으면 Spring이 HTTP 200 + JSON null 본문을 반환하는 것으로 통일.
    - 그 경우 resp.json()이 None이 되어 그대로 None을 반환한다.
    - 네트워크 오류/5xx 등 실제 장애만 502로 올린다.
    """
    url = f"{SPRING_BASE_URL}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=SPRING_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Spring 요청 실패 (%s): %s", url, e)
        raise HTTPException(status_code=502, detail="외부 데이터를 불러오지 못했습니다.")

    return resp.json()  # 본문이 null이면 파이썬 None


def get_user_goal(user_id: int) -> dict | None:
    """Spring: GET /chat/goal/{userId} -> {targetCalories, targetCarbsG, targetProteinG, targetFatG} | null"""
    data = spring_get(f"/chat/goal/{user_id}")
    if data is None:
        return None
    if data["targetCalories"] is None:
        return None
    return {
        "target_calories": data["targetCalories"],
        "target_carbs_g": data["targetCarbsG"],
        "target_protein_g": data["targetProteinG"],
        "target_fat_g": data["targetFatG"],
    }


def get_consumed_today(user_id: int) -> dict:
    """Spring: GET /chat/food-logs/{userId} -> {calories, carbsG, proteinG, fatG} | null"""
    data = spring_get(f"/chat/food-logs/{user_id}")
    if data is None:
        return {"calories": 0, "carbs_g": 0, "protein_g": 0, "fat_g": 0}
    return {
        "calories": data["calories"],
        "carbs_g": data["carbsG"],
        "protein_g": data["proteinG"],
        "fat_g": data["fatG"],
    }


def calc_remaining(goal: dict, consumed: dict) -> dict:
    return {
        "calories": max(0, goal["target_calories"] - consumed["calories"]),
        "carbs_g": max(0, goal["target_carbs_g"] - consumed["carbs_g"]),
        "protein_g": max(0, goal["target_protein_g"] - consumed["protein_g"]),
        "fat_g": max(0, goal["target_fat_g"] - consumed["fat_g"]),
    }


def query_food_candidates(remaining: dict, limit: int = 5) -> list[dict]:
    """
    Spring: GET /chat/foods?maxCalories={remaining_calories}
    -> [{foodName, servingGrams, calories, carbsG, proteinG, fatG}, ...] | null
    - 매크로(탄/단/지) 적합도 재정렬은 여기서 수행한다.
    """
    data = spring_get("/chat/foods", params={"maxCalories": remaining["calories"]})
    if not data:  # None 또는 빈 리스트
        return []

    candidates = [
        {
            "num": f["num"],
            "food_name": f["fname"],
            "serving_grams": f["unit"],
            "calories": f["kcal"],
            "carbs_g": f["carbs"],
            "protein_g": f["protein"],
            "fat_g": f["fat"],
        }
        for f in data
    ]

    def fit_score(food: dict) -> float:
        score = 0.0
        for key in ("carbs_g", "protein_g", "fat_g"):
            remain = remaining[key] or 1  # 0으로 나누기 방지
            over = max(0, food[key] - remain)
            score += (over / remain) ** 2
        calorie_fill_ratio = food["calories"] / (remaining["calories"] or 1)
        score += (1 - calorie_fill_ratio) ** 2
        return score

    candidates.sort(key=fit_score)
    return candidates[:limit]

# ---------------------------------------------------------------------------
# Tool 정의
# ---------------------------------------------------------------------------

@tool
def search_nutrition_guide(query: str) -> str:
    """
    영양학 가이드북(균형/절제/실천 관련 식품·영양 정보)을 검색합니다.

    특정 식품군의 영양성분, 권장 섭취량, 식습관 조언, 영양표시·소비기한,
    손씻기, 신체활동, 음주 등을 물을 때 사용하세요.

    query:
        검색할 영양 정보 질의문
    """
    chunks = search_chunks(query)

    logger.info(
        "tool=search_nutrition_guide query=%r top=%s",
        query,
        [(c["document_id"], round(c["similarity"], 4)) for c in chunks],
    )

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return (
            "[NO_MATCH] 가이드북에서 관련 문서를 찾지 못했습니다. "
            "이 사실을 사용자에게 언급하지 말고, 사용자의 질문에 대해 "
            "너의 일반적인 영양학 지식을 바탕으로 자연스럽게 답변해라."
        )

    return "\n\n".join(
        f"[{c['category']}/{c['subcategory']}] {c['content']}" for c in chunks
    )


@tool
def get_meal_recommendation(user_id: int) -> str:
    """
    사용자의 오늘 남은 칼로리/탄단지 예산에 맞는 식사 메뉴 후보를 조회합니다.

    사용자가 아침/점심/저녁 등 구체적인 메뉴·식단 추천을 요청할 때 사용하세요.

    user_id:
        추천을 요청한 사용자의 ID
    """
    goal = get_user_goal(user_id)
    if goal is None:
        return "설정된 목표 칼로리/영양소 정보가 없습니다. 먼저 목표를 설정해야 합니다."

    consumed = get_consumed_today(user_id)
    remaining = calc_remaining(goal, consumed)

    logger.info(
        "tool=get_meal_recommendation userId=%s consumed=%s remaining=%s",
        user_id, consumed, remaining,
    )

    foods = query_food_candidates(remaining)
    if not foods:
        return (
            f"오늘 남은 예산(칼로리 {remaining['calories']:.0f}kcal, "
            f"탄 {remaining['carbs_g']:.0f}g, 단 {remaining['protein_g']:.0f}g, "
            f"지 {remaining['fat_g']:.0f}g) 내에서 추천할 음식을 찾지 못했습니다."
        )

    remaining_text = (
        f"칼로리 {remaining['calories']:.0f}kcal, "
        f"탄수화물 {remaining['carbs_g']:.0f}g, "
        f"단백질 {remaining['protein_g']:.0f}g, "
        f"지방 {remaining['fat_g']:.0f}g"
    )
    foods_text = "\n".join(
        f"- {f['food_name']} ({f['serving_grams']}g, {f['calories']:.0f}kcal, "
        f"탄 {f['carbs_g']:.0f}g/단 {f['protein_g']:.0f}g/지 {f['fat_g']:.0f}g)"
        for f in foods
    )

    return f"[오늘 남은 영양소 예산]\n{remaining_text}\n\n[추천 음식 후보]\n{foods_text}"

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

# 챗봇용 종합

tools = [search_nutrition_guide, get_meal_recommendation]
tool_map = {t.name: t for t in tools}

llm_with_chat_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """당신은 체중관리 앱의 건강·영양 어시스턴트입니다.
다음 두 가지 도구를 상황에 맞게 사용하세요.

- search_nutrition_guide: 식품군의 영양성분, 권장 섭취량, 식습관 조언, 영양표시·소비기한,
  손씻기, 신체활동, 음주 등 영양학 가이드북 지식이 필요할 때 사용합니다.
- get_meal_recommendation: 사용자가 오늘 먹을 메뉴·식단 추천을 요청할 때 사용합니다.

[중요]
search_nutrition_guide 결과 가이드북에서 관련 정보를 찾지 못한 경우,
"찾지 못했습니다", "정보가 없습니다", "죄송하지만" 등 검색 실패를 알리는 표현을
절대 사용하지 마세요. 사용자에게는 검색 여부를 노출하지 말고, 마치 원래부터
알고 있던 일반적인 영양학 지식으로 자연스럽게 답변하세요.

도구 결과에 없는 내용을 답할 때도 사실과 다른 확정적 수치(정확한 g수, mg수 등)는
단정하지 말고 일반적인 수준에서 설명하세요."""


# 사진인식이나 음식,운동 db추가
llm_with_estimation_tools = llm.bind_tools([identify_food, estimate_food_nutrition, estimate_exercise_calorie])



# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()


class QueryRequest(BaseModel):
    userChat: str
    userId: int | None = None  # get_meal_recommendation 도구 사용 시 필요


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    user_chat = request.userChat.strip()

    if not user_chat:
        raise HTTPException(status_code=400, detail="userChat이 비어있습니다.")

    user_id_text = (
        f"현재 사용자 ID는 {request.userId}입니다. "
        f"get_meal_recommendation을 호출할 때 이 값을 user_id로 사용하세요."
        if request.userId is not None
        else "현재 사용자 ID가 없습니다. get_meal_recommendation은 호출할 수 없습니다."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{user_id_text}\n\n[사용자 질문]\n{user_chat}"),
    ]

    ai_message = llm_with_chat_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        # 도구 없이 바로 답할 수 있는 질문 (잡담, 인사 등)
        return QueryResponse(answer=ai_message.content)

    logger.info(
        "query=%r tool_calls=%s",
        user_chat,
        [tc["name"] for tc in ai_message.tool_calls],
    )

    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "get_meal_recommendation":
            if request.userId is None:
                messages.append(
                    ToolMessage(
                        content="사용자 식별 정보(userId)가 없어 식단을 추천할 수 없습니다.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            # LLM이 넘긴 user_id는 신뢰하지 않고, 실제 요청자의 ID로 강제 치환한다.
            tool_call["args"]["user_id"] = request.userId

        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    print("-------------------------tool_call :",tool_call,"--------------------")

    final_message = llm_with_chat_tools.invoke(messages)
    return QueryResponse(answer=final_message.content)





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

    response = llm_with_estimation_tools.invoke([message])

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

    response = llm_with_estimation_tools.invoke([message])

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

    response = llm_with_estimation_tools.invoke([message])

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