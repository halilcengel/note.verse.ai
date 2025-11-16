"""
Test script for the /embed endpoint

This script demonstrates how to use the embedding endpoint to upload PDF files
and store them in the Qdrant vector store.

Requirements:
- The FastAPI server should be running (uvicorn main:app --reload)
- Qdrant should be running on http://localhost:6333
- You need a PDF file to test with

Usage:
    python test_embed_endpoint.py <path_to_pdf_file> <collection_name>

Example:
    python test_embed_endpoint.py ./sample.pdf my_documents
"""

import requests
import sys
from datetime import datetime


def test_embed_endpoint(pdf_path: str, collection_name: str = "test_collection"):
    """
    Test the /embed endpoint by uploading a PDF file.

    Args:
        pdf_path: Path to the PDF file to upload
        collection_name: Name of the collection to store embeddings
    """
    url = "http://127.0.0.1:8000/embed"

    # Prepare the multipart form data
    files = {
        'file': open(pdf_path, 'rb')
    }

    data = {
        'collection_name': collection_name,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'createdBy': 'test_user'
    }

    try:
        print(f"Uploading {pdf_path} to collection '{collection_name}'...")
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success!")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
            print(f"  Collection: {result['collection_name']}")
            print(f"  Document chunks: {result['document_count']}")
            print(f"  Filename: {result['filename']}")
            print(f"  Created by: {result['created_by']}")
            print(f"  Date: {result['date']}")
            print(f"  Collection existed: {result['collection_existed']}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"  {response.json()}")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the server.")
        print("  Make sure the FastAPI server is running:")
        print("  uvicorn main:app --reload")
    except FileNotFoundError:
        print(f"\n✗ Error: File not found: {pdf_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    finally:
        files['file'].close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_embed_endpoint.py <pdf_path> [collection_name]")
        print("\nExample:")
        print("  python test_embed_endpoint.py ./sample.pdf my_documents")
        sys.exit(1)

    pdf_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "test_collection"

    test_embed_endpoint(pdf_path, collection_name)
