import json
import os

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "fitlog"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}


JSON_FILE = "nutrition_chunks.jsonl"


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def load_jsonl():

    data = []

    with open(JSON_FILE, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(f"{line_number}번째 줄 JSON 오류: {e}")

    return data


def insert_document(conn, item):

    document_id = item["id"]
    content = item["content"]

    metadata = item.get("metadata", {})

    source = metadata.get("source")
    category = metadata.get("category")
    subcategory = metadata.get("subcategory")
    section = metadata.get("section")
    doc_page = metadata.get("doc_page")

    # content → embedding
    embedding = embeddings.embed_query(content)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO nutrition_documents (
            document_id,
            content,
            source,
            category,
            subcategory,
            section,
            doc_page,
            metadata,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::jsonb,
            %s::vector
        )
        ON CONFLICT (document_id)
        DO UPDATE SET
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            category = EXCLUDED.category,
            subcategory = EXCLUDED.subcategory,
            section = EXCLUDED.section,
            doc_page = EXCLUDED.doc_page,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
        """,
        (
            document_id,
            content,
            source,
            category,
            subcategory,
            section,
            doc_page,
            json.dumps(metadata, ensure_ascii=False),
            embedding
        )
    )

    cursor.close()


def main():

    data = load_jsonl()

    print(f"총 {len(data)}개의 문서를 처리합니다.")

    conn = psycopg2.connect(**DB_CONFIG)

    try:

        for index, item in enumerate(data, start=1):

            insert_document(conn, item)

            print(
                f"[{index}/{len(data)}] "
                f"{item['id']} 저장 완료"
            )

        conn.commit()

        print()
        print("모든 문서 저장 완료!")

    except Exception as e:

        conn.rollback()

        print()
        print("오류 발생")
        print(e)

    finally:

        conn.close()


if __name__ == "__main__":
    main()