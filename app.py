import hashlib
import os
import shutil
from pathlib import Path
from typing import List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

APP_TITLE = "Thriving Serenity"
APP_SUBTITLE = "Decision Support per il Diritto Penale Italiano"
APP_CLIENT = "Created for Sonnet Malakaran"
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


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #f5f1e8;
        --panel: #fcfaf6;
        --panel-2: #f1eadf;
        --panel-3: #e8dfd1;
        --text: #1e2430;
        --muted: #5f6b7a;
        --soft: #7c8795;
        --accent: #8a6a2f;
        --accent-2: #234a6b;
        --danger-bg: #fff1f1;
        --danger-border: #e7b6b6;
        --danger-text: #8a3d3d;
        --border: rgba(30, 36, 48, 0.10);
        --shadow: 0 10px 30px rgba(36, 38, 44, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(138, 106, 47, 0.07), transparent 24%),
            radial-gradient(circle at top left, rgba(35, 74, 107, 0.06), transparent 20%),
            linear-gradient(180deg, #f8f4ec 0%, #f3ede3 100%);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eee6d8 0%, #e8dece 100%);
        border-right: 1px solid rgba(30, 36, 48, 0.08);
    }

    .main .block-container {
        padding-top: 2.1rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    .hero-wrap {
        margin-bottom: 1.1rem;
    }

    .legal-badge {
        display: inline-block;
        padding: 0.42rem 0.82rem;
        border-radius: 999px;
        background: rgba(138, 106, 47, 0.10);
        color: #6f5423;
        border: 1px solid rgba(138, 106, 47, 0.16);
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(241,234,223,0.92));
        border: 1px solid var(--border);
        padding: 1.65rem 1.6rem;
        border-radius: 22px;
        margin-bottom: 0.8rem;
        box-shadow: var(--shadow);
    }

    .hero-topline {
        color: var(--accent-2);
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.65rem;
    }

    .hero h1 {
        margin: 0;
        color: var(--text);
        font-size: 2.35rem;
        line-height: 1.12;
        letter-spacing: -0.02em;
    }

    .hero .subtitle {
        margin: 0.55rem 0 0 0;
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 900px;
    }

    .hero .client-line {
        margin-top: 1rem;
        display: inline-block;
        padding: 0.55rem 0.85rem;
        border-radius: 12px;
        background: rgba(35, 74, 107, 0.08);
        color: var(--accent-2);
        border: 1px solid rgba(35, 74, 107, 0.12);
        font-size: 0.98rem;
        font-weight: 700;
    }

    .intro-copy {
        color: var(--text);
        font-size: 1.05rem;
        line-height: 1.8;
        margin: 0.3rem 0 1.2rem 0;
        max-width: 920px;
    }

    .disclaimer-box {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
        color: var(--danger-text);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin: 0.9rem 0 1rem 0;
        line-height: 1.65;
    }

    .source-card {
        background: rgba(255,255,255,0.65);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 18px rgba(25, 28, 36, 0.04);
    }

    .source-title {
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }

    .source-meta {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.45rem;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .stMarkdown p,
    .stMarkdown li,
    .stChatMessage p,
    div[data-testid="stChatMessageContent"] p {
        font-size: 1.04rem;
        line-height: 1.8;
        color: var(--text);
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    h2, h3 {
        margin-top: 0.35rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(138, 106, 47, 0.25);
        background: linear-gradient(135deg, #8a6a2f, #6f5423);
        color: #fffdf8;
        font-weight: 700;
        min-height: 2.9rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(138, 106, 47, 0.36);
        filter: brightness(1.03);
    }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    div[data-testid="stFileUploader"] section {
        background-color: rgba(255,255,255,0.68) !important;
        color: var(--text) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(30, 36, 48, 0.12) !important;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.52);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.35rem 0.4rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.45);
        border-radius: 16px;
        border: 1px solid rgba(30, 36, 48, 0.08);
    }

    label, .stFileUploader label, .stTextInput label {
        color: var(--text) !important;
        font-weight: 600;
    }

    .stCaption, .st-emotion-cache-1wivap2 {
        color: var(--muted) !important;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 1.2rem 1rem;
        }

        .hero h1 {
            font-size: 1.85rem;
        }

        .hero .subtitle,
        .intro-copy,
        .stMarkdown p,
        .stMarkdown li {
            font-size: 1rem;
            line-height: 1.7;
        }

        .hero .client-line {
            width: 100%;
            text-align: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
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
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", None)
    return api_key, api_base


def get_embeddings():
    api_key, api_base = get_api_credentials()
    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovato.")
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAIEmbeddings(model="text-embedding-3-small", **kwargs)


def save_uploaded_files(uploaded_files) -> List[Path]:
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIRECTORY / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def load_pdf_documents(pdf_paths: List[Path]) -> List[Document]:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_collection_name(file_names: List[str]) -> str:
    joined = "|".join(sorted(file_names))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]
    return f"penalai_{digest}"


def clear_existing_collection(collection_name: str) -> None:
    collection_path = PERSIST_DIRECTORY / collection_name
    if collection_path.exists():
        shutil.rmtree(collection_path, ignore_errors=True)


def index_documents(uploaded_files) -> None:
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
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Documento sconosciuto")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(f"[Fonte: {source} - pagina {page}]\\n{doc.page_content}")
    return "\\n\\n".join(formatted_chunks)


def answer_question(question: str) -> tuple[str, List[Document]]:
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
    with st.sidebar:
        st.markdown(f"## ⚖️ {APP_TITLE}")
        st.caption(APP_SUBTITLE)
        st.markdown(
            f"<div class='small-note' style='margin-bottom:0.75rem;'><strong>{APP_CLIENT}</strong></div>",
            unsafe_allow_html=True,
        )
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
    st.markdown("<div class='hero-wrap'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='legal-badge'>Italian Criminal Law • AI Decision Support</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-topline">Legal Analysis Workspace</div>
            <h1>{APP_TITLE}</h1>
            <p class="subtitle">{APP_SUBTITLE}</p>
            <div class="client-line">{APP_CLIENT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='intro-copy'>Carica atti, sentenze, capi di imputazione o altri PDF processuali, indicizzali e poni domande in italiano o in inglese. L'interfaccia è ottimizzata per una lettura più chiara delle risposte e delle fonti.</div>",
        unsafe_allow_html=True,
    )


def render_upload_section() -> List:
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
    st.markdown("### Domande di esempio")
    cols = st.columns(3)
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        if cols[idx].button(question, use_container_width=True):
            st.session_state.pending_question = question


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)


def render_sources(docs: List[Document]) -> None:
    with st.expander("Sources", expanded=True):
        if not docs:
            st.info("Nessuna fonte disponibile per questa risposta.")
            return
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Documento sconosciuto")
            page = doc.metadata.get("page", "?")
            passage = doc.page_content.strip().replace("\\n", " ")
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
