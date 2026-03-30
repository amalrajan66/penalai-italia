import hashlib
import os
import shutil
from pathlib import Path
from typing import List

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

APP_TITLE = "PenalAI Italia"
APP_SUBTITLE = "Decision Support per il Diritto Penale Italiano"
DISCLAIMER = "For human review only - not legal advice"
PERSIST_DIRECTORY = Path("./chroma_db")
UPLOAD_DIRECTORY = Path("./uploaded_pdfs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT = """Sei un assistente legale specializzato nel diritto penale italiano.
Analizza i documenti forniti e rispondi in modo strutturato.
Indica sempre: 1) il problema giuridico principale, 2) gli articoli rilevanti del codice penale o di procedura penale, 3) i possibili percorsi procedurali, 4) le strategie possibili.
Ricorda: le tue risposte sono solo supporto alla decisione umana, non sostituiscono il giudizio del professionista legale."""
EXAMPLE_QUESTIONS = [
    "Quali sono le possibili strade procedurali in questo caso?",
    "Quali articoli del codice penale sono rilevanti?",
    "Quali strategie difensive emergono dagli atti?",
]


# Page configuration must be set before other Streamlit UI elements.
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Global styling for a polished dark theme suitable for legal demos.
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --panel: #121a2f;
        --panel-2: #18233f;
        --text: #e8ecf8;
        --muted: #aab4d0;
        --accent: #d4af37;
        --accent-2: #7aa2ff;
        --danger: #ff7b7b;
        --border: rgba(255, 255, 255, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(122,162,255,0.15), transparent 30%),
            radial-gradient(circle at top right, rgba(212,175,55,0.12), transparent 26%),
            linear-gradient(180deg, #0a0f1d 0%, #0d1426 100%);
        color: var(--text);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1830 0%, #0a1222 100%);
        border-right: 1px solid var(--border);
    }
    .hero {
        background: linear-gradient(135deg, rgba(212,175,55,0.18), rgba(122,162,255,0.12));
        border: 1px solid var(--border);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    }
    .hero h1 {
        margin: 0;
        color: var(--text);
        font-size: 2rem;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
        font-size: 1rem;
    }
    .legal-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(212,175,55,0.14);
        color: #f4d77a;
        border: 1px solid rgba(212,175,55,0.24);
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .disclaimer-box {
        background: rgba(255,123,123,0.08);
        border: 1px solid rgba(255,123,123,0.18);
        color: #ffd6d6;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0 1rem 0;
    }
    .source-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .source-title {
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .source-meta {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(212,175,55,0.3);
        background: linear-gradient(135deg, rgba(212,175,55,0.22), rgba(122,162,255,0.18));
        color: white;
        font-weight: 600;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: rgba(255,255,255,0.04);
        color: var(--text);
    }
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Initialize all session keys used by the app."""
    defaults = {
        "vectorstore": None,
        "retriever": None,
        "chat_history": [],
        "indexed_docs": [],
        "pending_question": None,
        "last_sources": [],
        "collection_name": None,
        "documents_indexed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_credentials() -> tuple[str, str | None]:
    """Read API credentials from environment variables (Railway-friendly)."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", None)
    return api_key, api_base


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local embedding model used for Chroma indexing."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def save_uploaded_files(uploaded_files) -> List[Path]:
    """Persist uploaded PDF files so LangChain loaders can process them."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIRECTORY / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def load_pdf_documents(pdf_paths: List[Path]) -> List[Document]:
    """Read all PDF pages and attach clean metadata for citations."""
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        documents.extend(pages)
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Chunk PDF text into citation-friendly passages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_collection_name(file_names: List[str]) -> str:
    """Create a deterministic Chroma collection name from the current upload set."""
    joined = "|".join(sorted(file_names))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]
    return f"penalai_{digest}"


def clear_existing_collection(collection_name: str) -> None:
    """Remove an old collection folder to allow clean re-indexing."""
    collection_path = PERSIST_DIRECTORY / collection_name
    if collection_path.exists():
        shutil.rmtree(collection_path, ignore_errors=True)


def index_documents(uploaded_files) -> None:
    """Save PDFs, parse them, chunk them, and persist a Chroma vector store."""
    if not uploaded_files:
        st.warning("Carica almeno un PDF prima di indicizzare.")
        return

    saved_paths = save_uploaded_files(uploaded_files)
    documents = load_pdf_documents(saved_paths)
    chunks = split_documents(documents)
    file_names = [path.name for path in saved_paths]
    collection_name = build_collection_name(file_names)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    st.session_state.indexed_docs = file_names
    st.session_state.collection_name = collection_name
    st.session_state.documents_indexed = True
    st.session_state.last_sources = []


def get_llm() -> ChatOpenAI:
    """Instantiate an OpenAI-compatible chat client using environment variables."""
    api_key, api_base = get_api_credentials()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY non trovato. Imposta la variabile d'ambiente nel tuo deployment."
        )

    client_kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
        "api_key": api_key,
    }
    if api_base:
        client_kwargs["base_url"] = api_base
    return ChatOpenAI(**client_kwargs)


def build_rag_prompt() -> ChatPromptTemplate:
    """Create the prompt template for structured, source-grounded legal answers."""
    template = """
{system_prompt}

Contesto documentale:
{context}

Domanda dell'utente:
{question}

Istruzioni aggiuntive:
- Rispondi nella lingua della domanda dell'utente.
- Mantieni una struttura chiara con titoli brevi e punti elenco quando utile.
- Se una conclusione non è supportata dagli atti, dichiaralo esplicitamente.
- Cita solo informazioni ricavabili dai documenti forniti.
- Chiudi con questa dicitura esatta: For human review only - not legal advice
"""
    return ChatPromptTemplate.from_template(template)


def format_context(docs: List[Document]) -> str:
    """Convert retrieved chunks into a context block for the model."""
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Documento sconosciuto")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(f"[Fonte: {source} - pagina {page}]\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def answer_question(question: str) -> tuple[str, List[Document]]:
    """Run retrieval and generate a grounded response with an OpenAI-compatible model."""
    if not st.session_state.retriever:
        raise ValueError("Indicizza prima i documenti per poter porre domande.")

    retrieved_docs = st.session_state.retriever.invoke(question)
    prompt = build_rag_prompt()
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "context": format_context(retrieved_docs),
            "question": question,
        }
    )
    return response.content, retrieved_docs


def render_sidebar() -> None:
    """Render the legal branding, disclaimer, and indexed document list."""
    with st.sidebar:
        st.markdown(f"## ⚖️ {APP_TITLE}")
        st.caption(APP_SUBTITLE)
        st.markdown(
            f"<div class='disclaimer-box'><strong>Disclaimer:</strong><br>{DISCLAIMER}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Documenti indicizzati")
        if st.session_state.indexed_docs:
            for name in st.session_state.indexed_docs:
                st.markdown(f"- {name}")
        else:
            st.markdown(
                "<span class='small-note'>Nessun documento indicizzato.</span>",
                unsafe_allow_html=True,
            )

        st.markdown("### Configurazione")
        st.markdown(
            "<span class='small-note'>API key letta dalle variabili d'ambiente del deployment.</span>",
            unsafe_allow_html=True,
        )


def render_header() -> None:
    """Render the top hero section and quick explanation."""
    st.markdown(
        "<div class='legal-badge'>Italian Criminal Law • AI Decision Support</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Carica atti, sentenze, capi di imputazione o altri PDF processuali, "
        "indicizzali e poni domande in italiano o in inglese."
    )


def render_upload_section() -> List:
    """Render uploader controls and return uploaded files."""
    st.markdown("### Documenti")
    uploaded_files = st.file_uploader(
        "Carica uno o più PDF del fascicolo",
        type=["pdf"],
        accept_multiple_files=True,
        help="Puoi caricare atti di indagine, imputazioni, verbali, memorie, ordinanze e sentenze.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Index Documents", use_container_width=True):
            with st.spinner("Indicizzazione dei documenti in corso..."):
                index_documents(uploaded_files)
            st.success("Documenti indicizzati con successo.")
    with col2:
        if st.button("Reset Session", use_container_width=True):
            for key in [
                "vectorstore",
                "retriever",
                "chat_history",
                "indexed_docs",
                "pending_question",
                "last_sources",
                "collection_name",
                "documents_indexed",
            ]:
                st.session_state[key] = (
                    [] if key in ["chat_history", "indexed_docs", "last_sources"] else None
                )
            st.session_state.documents_indexed = False
            st.success("Sessione azzerata.")
    return uploaded_files


def render_example_questions() -> None:
    """Render clickable question shortcuts for common legal workflows."""
    st.markdown("### Domande di esempio")
    cols = st.columns(3)
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        if cols[idx].button(question, use_container_width=True):
            st.session_state.pending_question = question


def render_chat_history() -> None:
    """Show all prior messages stored in session state."""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)


def render_sources(docs: List[Document]) -> None:
    """Render expandable source passages with exact page references."""
    with st.expander("Sources", expanded=True):
        if not docs:
            st.info("Nessuna fonte disponibile per questa risposta.")
            return
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Documento sconosciuto")
            page = doc.metadata.get("page", "?")
            passage = doc.page_content.strip().replace("\n", " ")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">Fonte {idx}: {source}</div>
                    <div class="source-meta">Pagina {page}</div>
                    <div>{passage}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def process_question(question: str) -> None:
    """Handle a user query, store messages, and display sources."""
    if not question:
        return

    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analisi dei documenti in corso..."):
                answer, docs = answer_question(question)
            st.markdown(
                f"<div class='disclaimer-box'><strong>Disclaimer:</strong> {DISCLAIMER}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(answer)
            render_sources(docs)
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.session_state.last_sources = docs
        except Exception as exc:
            error_message = f"Errore durante l'elaborazione: {exc}"
            st.error(error_message)
            st.session_state.chat_history.append(AIMessage(content=error_message))


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()
    render_sidebar()
    render_header()
    render_upload_section()
    render_example_questions()
    st.markdown("### Conversazione")
    render_chat_history()

    prompt_value = st.chat_input("Scrivi una domanda sui documenti caricati...")
    effective_question = st.session_state.pending_question or prompt_value

    if effective_question:
        st.session_state.pending_question = None
        process_question(effective_question)


if __name__ == "__main__":
    main()
