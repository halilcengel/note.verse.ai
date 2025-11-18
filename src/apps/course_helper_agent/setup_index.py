"""
Setup script to create the required index on course_id field in Qdrant.

This script creates a keyword index on the 'course_id' field which is required
for filtering documents by course.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

def setup_course_id_index(collection_name: str = "courses"):
    """
    Create a keyword index on the course_id field.

    Args:
        collection_name: Name of the Qdrant collection (default: "courses")
    """
    # Get Qdrant credentials from environment
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment")

    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    print(f"Creating index on 'course_id' field in collection '{collection_name}'...")

    try:
        # Create keyword index on course_id field
        client.create_payload_index(
            collection_name=collection_name,
            field_name="course_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"✓ Successfully created keyword index on 'course_id' field")

    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓ Index on 'course_id' already exists")
        else:
            print(f"✗ Error creating index: {str(e)}")
            raise

    # Verify the index was created
    try:
        collection_info = client.get_collection(collection_name)
        print(f"\nCollection info:")
        print(f"  - Collection: {collection_name}")
        print(f"  - Points count: {collection_info.points_count}")
        print(f"  - Payload schema: {collection_info.payload_schema}")

    except Exception as e:
        print(f"Warning: Could not retrieve collection info: {str(e)}")

    print(f"\n✓ Setup complete! You can now filter by course_id.")


def create_additional_indexes(collection_name: str = "courses"):
    """
    Create additional useful indexes for better query performance.

    Args:
        collection_name: Name of the Qdrant collection (default: "courses")
    """
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    indexes_to_create = [
        ("document-id", PayloadSchemaType.KEYWORD),
        ("title", PayloadSchemaType.TEXT),
        ("uploaded_by", PayloadSchemaType.KEYWORD),
    ]

    print(f"\nCreating additional indexes on collection '{collection_name}'...")

    for field_name, schema_type in indexes_to_create:
        try:
            if schema_type == PayloadSchemaType.TEXT:
                # For text fields, use text index with tokenization
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=TextIndexParams(
                        type="text",
                        tokenizer=TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    )
                )
            else:
                # For keyword fields
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )

            print(f"✓ Created {schema_type} index on '{field_name}'")

        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  - Index on '{field_name}' already exists")
            else:
                print(f"✗ Error creating index on '{field_name}': {str(e)}")

    print("\n✓ All indexes created!")


if __name__ == "__main__":
    # Create the required course_id index
    setup_course_id_index()

    # Create additional helpful indexes
    create_additional_indexes()
