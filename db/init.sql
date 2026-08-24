CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE nutrition_documents (
    id BIGSERIAL PRIMARY KEY,

    document_id VARCHAR(200) UNIQUE NOT NULL,

    content TEXT NOT NULL,

    source VARCHAR(300),

    category VARCHAR(100),

    subcategory VARCHAR(200),

    section VARCHAR(300),

    doc_page INTEGER,

    metadata JSONB,

    embedding VECTOR(1536),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);