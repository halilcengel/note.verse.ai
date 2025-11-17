import os

import requests
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

from settings import settings


def download_pdf(url, temp_path):
    """Download PDF from URL to temporary file"""
    response = requests.get(url, stream=True ,verify=False)
    response.raise_for_status()

    with open(temp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return temp_path


def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load PDF and split into chunks"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)
    return splits


def add_pdfs_to_vectorstore(pdf_urls, collection_name="school_data",
                            qdrant_url="http://localhost:6333",
                            chunk_size=1000, chunk_overlap=200,
                            openai_api_key=None):
    """
    Download PDFs from URLs and add them to Qdrant vector store

    Args:
        pdf_urls: List of PDF URLs to process
        collection_name: Name of the Qdrant collection
        qdrant_url: URL of the Qdrant instance
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        openai_api_key: OpenAI API key (optional, will use env var if not provided)
    """
    # Initialize embeddings and vector store
    # Strip whitespace from API key if provided
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_api_key
        )
    else:
        # If using environment variable, strip it
        import os
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    all_documents = []

    # Process each PDF URL
    for url in pdf_urls:
        print(f"Processing: {url}")

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                temp_path = tmp_file.name

            # Download PDF
            print("  Downloading...")
            download_pdf(url, temp_path)

            # Load and split PDF
            print("  Loading and splitting...")
            documents = load_and_split_pdf(temp_path, chunk_size, chunk_overlap)

            # Add source URL to metadata
            for doc in documents:
                doc.metadata['source_url'] = url

            all_documents.extend(documents)
            print(f"  ✓ Added {len(documents)} chunks from {url}")

        except Exception as e:
            print(f"  ✗ Error processing {url}: {str(e)}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Add all documents to vector store
    if all_documents:
        print(f"\nAdding {len(all_documents)} total chunks to vector store...")
        qdrant.add_documents(all_documents)
        print("✓ Successfully added all documents to Qdrant!")
    else:
        print("No documents to add.")

    return len(all_documents)


# Example usage
if __name__ == "__main__":
    # List your PDF URLs here
    pdf_urls = [
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/Mazeretlerin_Kabul%C3%BCne_ve_Mazeret.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/Muafiyet_ve_%C4%B0ntibak_%C4%B0%C5%9Flemleri_Y%C3%B6.2019%20tarihli%20ve%2023-03%20say%C4%B1l%C4%B1%20senato%20karar%C4%B1)===.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Diploma_Diploma_Eki_ve_Di%C4%9Fer_Be%20-%20Copy%201.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/%C3%96zel_%C3%96%C4%9Frenci_Y%C3%B6nergesi_13_03.2019%20tarihli%20ve%2004-03%20say%C4%B1l%C4%B1%20senato%20karar%C4%B1).pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/%C3%87ift_Anadal_ve_Yan_dal_Programla%20-%20Copy%201.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Uluslararas%C4%B1_%C3%96%C4%9Frencileri_Y%C3%B6nerge.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/Uygulamal%C4%B1_E%C4%9Fitim_Y%C3%B6nergesi_07_.2021%20tarihli%20ve%2016-01%20say%C4%B1l%C4%B1%20senato%20karar%C4%B1).pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/UOP/Dilmer_Y%C3%B6nergesi.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/e%C4%9Fitim_komisyonu_y%C3%B6nergesi_10_09.2025%20(1).pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/Y%C3%96NERGELER/Ortak_Dersler_Koordinat%C3%B6rl%C3%BC%C4%9F%C3%BC_Y%C3%B6.2025%20senato%20karar%C4%B1).pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/Dokumanlar/oidb/%C3%96l%C3%A7me_De%C4%9Ferlendirme_Esaslar%C4%B1_202.pdf",
        "https://oidb.bakircay.edu.tr/Yuklenenler/NOT_DONUSUM_TABLOSU.pdf"
    ]

    qdrant_url = os.getenv("QDRANT_URL", "").strip()

    total_chunks = add_pdfs_to_vectorstore(
        pdf_urls=pdf_urls,
        collection_name="school_data",
        qdrant_url=qdrant_url,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"\nTotal chunks added: {total_chunks}")