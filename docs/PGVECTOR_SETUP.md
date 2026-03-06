# pgvector Local Setup Guide

pgvector is a PostgreSQL extension for vector similarity search.
It can be used as a vector DB backend for the RAG pipeline.

## Installation

### Option 1: Docker (Recommended)

```bash
# Start pgvector container
docker run -d --name ai-pgvector \
  -e POSTGRES_USER=ai_assistant \
  -e POSTGRES_PASSWORD=ai_assistant_dev \
  -e POSTGRES_DB=ai_vectors \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Or use docker-compose (from project root)
docker compose --profile pgvector up -d
```

### Option 2: Native Linux (Debian/Ubuntu)

```bash
# Install PostgreSQL 16 + pgvector
sudo apt install postgresql-16 postgresql-16-pgvector

# Start PostgreSQL
sudo systemctl start postgresql
```

### Option 3: macOS (Homebrew)

```bash
brew install postgresql@16
brew install pgvector

brew services start postgresql@16
```

### Option 4: Windows

1. Install PostgreSQL 16 from https://www.postgresql.org/download/windows/
2. Download pgvector from https://github.com/pgvector/pgvector/releases
3. Copy `vector.dll` to PostgreSQL's `lib` directory
4. Copy `vector.control` and SQL files to `share/extension`

## Database Setup

```sql
-- Connect to PostgreSQL
psql -U ai_assistant -d ai_vectors

-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the vectors table
CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    collection VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- OpenAI ada-002 dimension
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings(collection);

-- IVFFlat index for approximate nearest neighbor search
-- Adjust lists based on data size: sqrt(num_rows) is a good starting point
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- For exact search on smaller datasets, use HNSW instead:
-- CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
--     ON embeddings USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
```

## Configuration

In your `config.json`:

```json
{
  "vector_db": {
    "backend": "pgvector",
    "pgvector": {
      "connection_string": "postgresql://ai_assistant:ai_assistant_dev@localhost:5432/ai_vectors",
      "table_name": "embeddings",
      "embedding_dimension": 1536
    }
  }
}
```

## Verify Installation

```sql
-- Check pgvector is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check version
SELECT extversion FROM pg_extension WHERE extname = 'vector';

-- Test vector operations
SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector AS cosine_distance;
```

## Troubleshooting

**"could not open extension control file"**
- pgvector extension files not installed. Re-install pgvector for your PostgreSQL version.

**"type vector does not exist"**
- Run `CREATE EXTENSION vector;` in the target database.

**Connection refused**
- Ensure PostgreSQL is running: `pg_isready -h localhost -p 5432`
- Check `pg_hba.conf` allows local connections.

**Slow queries**
- Ensure IVFFlat or HNSW index exists on the vector column.
- For IVFFlat, increase `probes` for better recall: `SET ivfflat.probes = 10;`
- For HNSW, increase `ef_search`: `SET hnsw.ef_search = 100;`
