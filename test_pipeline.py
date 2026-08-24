"""
파이프라인을 아래에서 위로 단계별 검증한다.
  1단계: Spring API가 응답하는가 (LLM/FastAPI 로직 없이)
  2단계: FastAPI의 데이터 조회 함수들이 Spring 응답을 올바르게 파싱하는가
  3단계: tool 함수가 LLM 없이 단독으로 잘 동작하는가
  4단계: 전체 /query 엔드포인트가 의도한 라우팅(RAG/MEAL/잡담)으로 잘 흘러가는가

사용법:
  python test_pipeline.py 1   # 1단계만 실행
  python test_pipeline.py     # 전체 실행

주의: 2~3단계는 실제 존재하는 userId로 바꿔서 테스트해야 의미가 있습니다.
"""

import sys

import httpx

SPRING_BASE_URL = "http://localhost:8070"
FASTAPI_BASE_URL = "http://localhost:8000"
TEST_USER_ID = 2  # 실제 DB에 있는 userId로 바꿔서 테스트하세요


def section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1단계: Spring API 자체가 잘 응답하는지 (curl로 해도 되지만 파이썬으로 일괄 확인)
# ---------------------------------------------------------------------------
def step1_spring_raw():
    section("1단계: Spring API 원본 응답 확인")

    endpoints = [
        ("GET", f"/chat/goal/{TEST_USER_ID}"),
        ("GET", f"/chat/food-logs/{TEST_USER_ID}"),
        ("GET", "/chat/foods", {"maxCalories": 500}),
    ]

    for method, path, *params in endpoints:
        p = params[0] if params else None
        url = f"{SPRING_BASE_URL}{path}"
        try:
            resp = httpx.get(url, params=p, timeout=5.0)
            print(f"[{resp.status_code}] {method} {path} params={p}")
            print(f"  body: {resp.text[:300]}")
        except httpx.HTTPError as e:
            print(f"  요청 실패: {e}")


# ---------------------------------------------------------------------------
# 2단계: FastAPI 데이터 조회 함수가 Spring 응답을 올바르게 파싱하는지
#         (main.py를 import해서 함수만 직접 호출, LLM/엔드포인트는 건드리지 않음)
# ---------------------------------------------------------------------------
def step2_data_functions():
    section("2단계: FastAPI 데이터 조회 함수 단위 테스트")

    from main import (
        calc_remaining,
        get_consumed_today,
        get_user_goal,
        query_food_candidates,
    )

    goal = get_user_goal(TEST_USER_ID)
    print(f"get_user_goal({TEST_USER_ID}) -> {goal}")
    assert goal is None or set(goal.keys()) == {
        "target_calories", "target_carbs_g", "target_protein_g", "target_fat_g"
    }, "goal 딕셔너리 키가 예상과 다릅니다"

    consumed = get_consumed_today(TEST_USER_ID)
    print(f"get_consumed_today({TEST_USER_ID}) -> {consumed}")
    assert set(consumed.keys()) == {
        "calories", "carbs_g", "protein_g", "fat_g"
    }, "consumed 딕셔너리 키가 예상과 다릅니다"

    if goal is not None:
        remaining = calc_remaining(goal, consumed)
        print(f"calc_remaining -> {remaining}")

        foods = query_food_candidates(remaining)
        print(f"query_food_candidates -> {len(foods)}건")
        for f in foods:
            print(f"  - {f}")
    else:
        print("goal이 None이라 remaining/foods 계산은 건너뜁니다. "
              "(Spring에 해당 userId의 목표 데이터가 없다는 뜻)")


# ---------------------------------------------------------------------------
# 3단계: tool 함수를 LLM 없이 직접 호출 (도구 로직 자체 검증)
# ---------------------------------------------------------------------------
def step3_tools_directly():
    section("3단계: tool 함수 단독 실행 (LLM 우회)")

    from main import get_meal_recommendation, search_nutrition_guide

    print("--- search_nutrition_guide ---")
    result = search_nutrition_guide.invoke({"query": "단백질은 왜 필요한가요?"})
    print(result[:300], "...\n")

    print("--- get_meal_recommendation ---")
    result = get_meal_recommendation.invoke({"user_id": TEST_USER_ID})
    print(result)


# ---------------------------------------------------------------------------
# 4단계: 전체 /query 엔드포인트 (서버가 떠 있어야 함: uvicorn main:app)
# ---------------------------------------------------------------------------
def step4_full_endpoint():
    section("4단계: 전체 /query 엔드포인트 (라우팅 확인)")

    cases = [
        {"userChat": "단백질은 왜 필요한가요?", "userId": TEST_USER_ID},   # RAG 기대
        {"userChat": "저녁 뭐 먹을까?", "userId": TEST_USER_ID},           # MEAL 기대
        {"userChat": "안녕!", "userId": TEST_USER_ID},                     # 잡담 기대
        {"userChat": "저녁 뭐 먹을까?", "userId": None},                   # userId 없이 MEAL 요청 (예외 처리 확인)
    ]

    for case in cases:
        try:
            resp = httpx.post(f"{FASTAPI_BASE_URL}/query", json=case, timeout=30.0)
            print(f"\n요청: {case}")
            print(f"[{resp.status_code}] {resp.json()}")
        except httpx.HTTPError as e:
            print(f"요청 실패: {e}")


if __name__ == "__main__":
    steps = {
        "1": step1_spring_raw,
        "2": step2_data_functions,
        "3": step3_tools_directly,
        "4": step4_full_endpoint,
    }

    if len(sys.argv) > 1 and sys.argv[1] in steps:
        steps[sys.argv[1]]()
    else:
        for fn in steps.values():
            fn()
