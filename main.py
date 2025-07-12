import faiss
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore


def replace_tabs_with_spaces(documents: List) -> List:
    """
    Replaces tab characters with spaces in the page content of each document.

    Args:
        documents (List): A list of documents with 'page_content' attributes.

    Returns:
        List: The modified list with tabs replaced.
    """
    for doc in documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return documents


# ✅ 初始化一次 embedding 模型，后续传入使用
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")


def generate_hypothetical_prompt_embeddings(chunk_text: str, embedding_model: HuggingFaceEmbeddings) -> Tuple[str, List[List[float]]]:
    """
    Generate and embed hypothetical questions for a chunk of text.

    Args:
        chunk_text (str): The chunk text to process.
        embedding_model (HuggingFaceEmbeddings): The embedding model instance.

    Returns:
        Tuple[str, List[List[float]]]: The original chunk and its question embeddings.
    """
    llm = ChatGroq(model_name="qwen/qwen3-32b", groq_api_key="gsk_xxx")

    prompt = PromptTemplate.from_template(
        "Analyze the following text and generate questions that reflect its core content. "
        "Each question should be a single line, no numbering or bullet points.\n\n"
        "Text:\n{chunk_text}\n\nQuestions:\n"
    )

    question_chain = prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")
    embeddings = embedding_model.embed_documents(questions)

    return chunk_text, embeddings


def prepare_vector_store(chunks: List[str], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Build FAISS vector store from text chunks and precomputed question embeddings.

    Args:
        chunks (List[str]): Text chunks.
        embedding_model (HuggingFaceEmbeddings): Shared embedding model.

    Returns:
        FAISS: The final vector store.
    """
    vector_store = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(generate_hypothetical_prompt_embeddings, chunk, embedding_model) for chunk in chunks]

        for f in tqdm(as_completed(futures), total=len(chunks)):
            chunk, vectors = f.result()

            if vector_store is None:
                vector_store = FAISS(embedding_function=embedding_model, index=faiss.IndexFlatL2(len(vectors[0])), docstore=InMemoryDocstore(), index_to_docstore_id={})

            vector_store.add_embeddings([(chunk, vec) for vec in vectors])

    return vector_store


def encode_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    Loads and encodes a PDF into a vector store using hypothetical prompt embeddings.

    Args:
        path (str): Path to the PDF file.
        chunk_size (int): Number of characters per chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        FAISS: Vector store with embedded chunks.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Clean and chunk the text
    cleaned_documents = replace_tabs_with_spaces(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = [chunk.page_content for chunk in splitter.split_documents(cleaned_documents)]

    return prepare_vector_store(chunks, embedding_model)


if __name__ == "__main__":
    # Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    PDF_PATH = "/Users/hzq/py_projects/hype-rag/Understanding_Climate_Change.pdf"
    TEST_QUERY = "What is the main cause of climate change?"

    # Step 1: Process PDF into vector store
    vector_store = encode_pdf(PDF_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Step 2: Convert vector store to retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 3: Create LLM for answering
    llm = ChatGroq(model_name="qwen/qwen3-32b", groq_api_key="gsk_xxx")

    # Step 4: Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Step 5: Ask question
    print(f"Question: {TEST_QUERY}")
    answer = qa_chain.run(TEST_QUERY)

    # Step 6: Output result
    print(f"\nAnswer:\n{answer}")
