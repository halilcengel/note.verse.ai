import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    TextIndexParams,
    TokenizerType,
    PayloadSchemaType,
)

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collection_name = "courses"

print(f"Creating collection '{collection_name}'...")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=3072,
        distance=Distance.COSINE,
        on_disk=False,
    ),
    hnsw_config={
        "m": 24,
        "ef_construct": 256,
    },
    on_disk_payload=True,
)

print(f"✓ Collection '{collection_name}' created successfully")

print("\nCreating payload indexes...")

print("  - Creating index for metadata.course_id (keyword)...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.course_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

print("  - Creating index for metadata.title (text)...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.title",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WHITESPACE,
        lowercase=True,
    ),
)

print("  - Creating index for metadata.document-id (keyword)...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.document-id",
    field_schema=PayloadSchemaType.KEYWORD,
)

print("  - Creating index for metadata.uploaded_by (keyword)...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.uploaded_by",
    field_schema=PayloadSchemaType.KEYWORD,
)

print("  - Creating index for metadata.uploaded_at (datetime)...")
client.create_payload_index(
    collection_name=collection_name,
    field_name="metadata.uploaded_at",
    field_schema=PayloadSchemaType.DATETIME,
)

print("\n✓ All indexes created successfully")

print(f"\nCollection '{collection_name}' information:")
collection_info = client.get_collection(collection_name=collection_name)
print(f"  - Vectors: {collection_info.vectors_count}")
print(f"  - Points: {collection_info.points_count}")
print(f"  - Status: {collection_info.status}")

