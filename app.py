import os
import time
import io
import sys
import importlib
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Hot-reload safety check for legal_orchestrator in persistent Streamlit runtime
try:
    if "legal_orchestrator" in sys.modules:
        import legal_orchestrator
        if not hasattr(legal_orchestrator.LegalOrchestrator, "classify_domain"):
            importlib.reload(legal_orchestrator)
            legal_orchestrator._global_orchestrator = None
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="LegalRAG Assistant | Enterprise Counsel & Due Diligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for legal aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    
    /* Workspace banner cards */
    .workspace-banner {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 1.2rem;
    }
    
    .workspace-banner-sample {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 1.2rem;
    }
    
    /* Citation card styling */
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        border-left: 4px solid #2563EB;
    }
    
    .citation-header {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 4px;
    }
    
    .citation-meta {
        font-size: 0.78rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
    }
    
    .citation-text {
        font-size: 0.84rem;
        line-height: 1.55;
        color: #334155;
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #F1F5F9;
        white-space: pre-wrap;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 6px;
    }
    
    .badge-user {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    
    .badge-contracts {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    
    .badge-const {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    
    .badge-latency {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def get_catalog_data() -> dict:
    """Fetch indexed catalog from API or in-process orchestrator."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/documents", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    from legal_orchestrator import get_orchestrator
    return get_orchestrator().get_catalog()


def query_backend(query: str, domain: str, doc_filter: str = None) -> dict:
    """Send query to FastAPI backend or fallback directly to in-process orchestrator."""
    payload = {
        "query": query,
        "domain": domain,
        "doc_filter": doc_filter if doc_filter and doc_filter not in ["All Uploaded Documents", "All Contracts", "All Sources"] else None
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/api/query", json=payload, timeout=90)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Direct in-process fallback
    import legal_orchestrator
    if not hasattr(legal_orchestrator.LegalOrchestrator, "classify_domain"):
        importlib.reload(legal_orchestrator)
        legal_orchestrator._global_orchestrator = None

    orchestrator = legal_orchestrator.get_orchestrator()
    if not hasattr(orchestrator, "classify_domain"):
        importlib.reload(legal_orchestrator)
        legal_orchestrator._global_orchestrator = None
        orchestrator = legal_orchestrator.get_orchestrator()

    if not hasattr(orchestrator, "classify_domain"):
        orchestrator.classify_domain = legal_orchestrator.LegalOrchestrator.classify_domain.__get__(orchestrator)

    res = orchestrator.query(
        query_text=query,
        domain=domain,
        doc_filter=payload["doc_filter"]
    )
    return {
        "query": res.query,
        "answer": res.answer,
        "domain": res.domain,
        "doc_filter": res.doc_filter,
        "citations": res.citations,
        "execution_time_seconds": res.execution_time_seconds,
        "model": res.model,
        "embedding_model": res.embedding_model,
        "reranker_model": res.reranker_model
    }


def upload_user_files(uploaded_files, progress_bar, status_text) -> dict:
    """Upload and index files via FastAPI or in-process fallback."""
    file_tuples = []
    for f in uploaded_files:
        content = f.getvalue()
        file_tuples.append((f.name, content))

    # Attempt API upload first
    try:
        multipart_files = [
            ("files", (f.name, f.getvalue())) for f in uploaded_files
        ]
        status_text.text("Sending files to backend API...")
        resp = requests.post(f"{API_BASE_URL}/api/upload", files=multipart_files, timeout=300)
        if resp.status_code == 200:
            progress_bar.progress(1.0)
            status_text.text("Corpus indexing complete!")
            return resp.json()
    except Exception:
        pass

    # In-process fallback with real-time progress callbacks
    from legal_orchestrator import get_orchestrator
    orchestrator = get_orchestrator()

    def update_progress(percent: float, msg: str):
        progress_bar.progress(min(max(percent, 0.0), 1.0))
        status_text.text(msg)

    return orchestrator.ingest_user_documents(file_tuples, progress_callback=update_progress)


def clear_user_corpus_backend():
    """Clear active user corpus in backend or in-process."""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/clear-user-corpus", timeout=10)
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    from legal_orchestrator import get_orchestrator
    get_orchestrator().clear_user_corpus()
    return True


# Fetch catalog to check current state
catalog = get_catalog_data()
user_corpus_meta = catalog.get("user_corpus", {})
user_docs = user_corpus_meta.get("documents", [])
user_corpus_active = user_corpus_meta.get("is_active", False)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales.png", width=46)
    st.title("LegalRAG Workspace")
    st.caption("AI Legal Intelligence & Due Diligence")

    st.markdown("---")
    st.subheader("📂 Active Corpus Mode")
    
    # Mode selector: Primary is User Uploaded Corpus; Secondary is Sample/Demo
    mode_options = [
        "📁 My Uploaded Corpus (Primary)",
        "⚖️ Sample Pre-loaded Corpus (Demo)"
    ]
    workspace_mode = st.radio(
        "Choose Corpus Workspace:",
        options=mode_options,
        index=0,
        help="Select whether to analyze your personal uploaded legal documents or test with pre-loaded sample materials."
    )

    selected_doc_filter = None
    active_domain_key = "user_corpus"

    # =========================================================
    # WORKSPACE 1: USER UPLOADED CORPUS (PRIMARY)
    # =========================================================
    if workspace_mode == "📁 My Uploaded Corpus (Primary)":
        active_domain_key = "user_corpus"
        st.markdown("---")
        st.subheader("📤 Upload Your Documents")
        st.caption("Upload commercial contracts, legal briefs, statutes, or regulatory PDFs/TXT/MD.")

        uploaded_files = st.file_uploader(
            "Select legal files:",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Supports searchable PDFs and clean text agreements. Chunks are parsed hierarchically by clause."
        )

        if uploaded_files:
            if st.button("⚡ Process & Index Corpus", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                with st.spinner("Processing legal documents..."):
                    try:
                        res = upload_user_files(uploaded_files, progress_bar, status_text)
                        st.success(f"✅ Indexed {res.get('total_documents', len(uploaded_files))} document(s) into {res.get('total_chunks', 0)} legal chunks!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")

        # Display active user corpus stats
        if user_corpus_active and user_docs:
            st.markdown("---")
            st.subheader("📋 Active User Documents")
            st.caption(f"**{len(user_docs)} files** indexed ({user_corpus_meta.get('total_chunks', 0)} chunks, {user_corpus_meta.get('parent_sections', 0)} parent sections)")
            
            selected_doc_filter = st.selectbox(
                "Filter query scope:",
                options=["All Uploaded Documents"] + user_docs,
                index=0,
                help="Scope retrieval specifically to one of your uploaded agreements or documents."
            )

            if st.button("🗑️ Reset / Clear Uploaded Corpus"):
                clear_user_corpus_backend()
                st.success("Uploaded corpus cleared.")
                time.sleep(0.5)
                st.rerun()
        else:
            st.info("ℹ️ No personal documents uploaded yet. Upload your agreements or switch to Sample Corpus below.")

    # =========================================================
    # WORKSPACE 2: SAMPLE / DEMO CORPUS
    # =========================================================
    else:
        st.markdown("---")
        st.subheader("⚖️ Sample Demo Practice Area")
        st.caption("Explore pre-indexed benchmark datasets.")

        sample_choice = st.radio(
            "Select Sample Domain:",
            options=["Auto-Detect (Agentic Routing)", "Commercial Contracts Due Diligence", "Indian Constitutional Law"],
            index=0
        )

        if sample_choice == "Commercial Contracts Due Diligence":
            active_domain_key = "commercial_contracts"
        elif sample_choice == "Indian Constitutional Law":
            active_domain_key = "constitutional_law"
        else:
            active_domain_key = "auto"

        # Scope filter for sample commercial contracts
        if active_domain_key in ["commercial_contracts", "auto"]:
            st.markdown("---")
            st.subheader("📄 Scope Sample Contract")
            sample_contracts = catalog.get("commercial_contracts", {}).get("documents", [])
            selected_doc_filter = st.selectbox(
                "Filter to specific sample agreement:",
                options=["All Contracts"] + sample_contracts,
                index=0,
                help="Scope vector search to a specific sample contract."
            )

    st.markdown("---")
    st.subheader("🛡️ Production Architecture")
    st.markdown("""
    - **LLM**: `gemini-3.5-flash-lite`
    - **Embeddings**: `gemini-embedding-2` (3072d)
    - **Reranker**: `ms-marco-MiniLM-L-12-v2`
    - **Retrieval**: Dense Qdrant + BM25 Lexical
    - **Expansion**: Small-to-Big Sibling Windows
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main Workspace Header & Banners ---
st.markdown('<div class="main-header">⚖️ LegalRAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise legal intelligence with clause-aware chunking, hybrid retrieval, and verifiable source citations.</div>', unsafe_allow_html=True)

# Workspace Banner
if workspace_mode == "📁 My Uploaded Corpus (Primary)":
    if user_corpus_active and user_docs:
        docs_badge_list = " &bull; ".join([f"<code>{d}</code>" for d in user_docs[:3]])
        if len(user_docs) > 3:
            docs_badge_list += f" and {len(user_docs) - 3} more"
        st.markdown(f"""
        <div class="workspace-banner">
            <strong>📁 Active Workspace: Personal Legal Corpus</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">
                Searching across {len(user_docs)} uploaded file(s): {docs_badge_list} ({user_corpus_meta.get('total_chunks', 0)} total chunks).
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="workspace-banner">
            <strong>📁 Active Workspace: Personal Legal Corpus</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">
                Upload your agreements, NDAs, or legal filings using the sidebar to analyze your own corpus. Alternatively, switch to the <strong>Sample Pre-loaded Corpus</strong> in the sidebar to test demo contracts.
            </span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="workspace-banner-sample">
        <strong>⚖️ Active Workspace: Pre-loaded Sample Corpus (Demo)</strong><br>
        <span style="font-size: 0.85rem; color: #475569;">
            Demonstrating dual-domain retrieval across 5 commercial SEC agreements and Indian Constitutional Law acts.
        </span>
    </div>
    """, unsafe_allow_html=True)

# --- Quick Query Pills ---
st.markdown("**Quick Queries:**")
pill_cols = st.columns(4 if workspace_mode == "📁 My Uploaded Corpus (Primary)" else 3)

if workspace_mode == "📁 My Uploaded Corpus (Primary)":
    with pill_cols[0]:
        if st.button("📋 Key Obligations"):
            st.session_state.user_query = "What are the core obligations, deliverables, and commitments of the parties under this agreement?"
    with pill_cols[1]:
        if st.button("⏳ Expiration & Term"):
            st.session_state.user_query = "What is the initial term, expiration date, or duration of this agreement, and how can it be renewed or terminated?"
    with pill_cols[2]:
        if st.button("🛡️ Liability Caps"):
            st.session_state.user_query = "What limitations of liability, indemnification obligations, or monetary caps are established in the agreement?"
    with pill_cols[3]:
        if st.button("⚖️ Governing Law"):
            st.session_state.user_query = "What state or governing law applies to the agreement, and what is the specified dispute resolution venue?"
else:
    with pill_cols[0]:
        if st.button("🏛️ Article 21 & Maneka Gandhi"):
            st.session_state.user_query = "What did the Supreme Court hold regarding the procedure established by law in the Maneka Gandhi case?"
    with pill_cols[1]:
        if st.button("📜 LimeEnergy Governing Law"):
            st.session_state.user_query = "In the LIMEENERGYCO Distributor Agreement, what state law governs the interpretation of the contract?"
    with pill_cols[2]:
        if st.button("⏳ BorrowMoney Expiration Date"):
            st.session_state.user_query = "In the BORROWMONEYCOM Joint Venture Agreement, on what date will the contract's initial term expire?"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            dom = meta.get("domain", "")
            if dom == "user_corpus":
                dom_badge = "badge-user"
                dom_label = "Personal Uploaded Corpus"
            elif dom == "commercial_contracts":
                dom_badge = "badge-contracts"
                dom_label = "Commercial Contracts (Sample)"
            else:
                dom_badge = "badge-const"
                dom_label = "Constitutional Law (Sample)"

            st.markdown(
                f'<span class="badge-pill {dom_badge}">🏛️ {dom_label}</span>'
                f'<span class="badge-pill badge-latency">⏱️ {meta.get("latency")}s</span>',
                unsafe_allow_html=True
            )
        if "citations" in msg and msg["citations"]:
            with st.expander(f"📚 Verifiable Source Citations ({len(msg['citations'])} sources)", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">Rank #{cit['rank']}: {cit['section']}</div>
                        <div class="citation-meta">Document: {cit['document_name']} | Reference: {cit['citation_id']}</div>
                        <div class="citation-text">{cit['excerpt']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input prompt
user_input = st.chat_input("Ask a legal question, contract clause query, or due diligence matter...")
if "user_query" in st.session_state and st.session_state.user_query:
    user_input = st.session_state.user_query
    st.session_state.user_query = None

if user_input:
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant thinking & query execution
    with st.chat_message("assistant"):
        spinner_msg = (
            "Analyzing personal uploaded legal corpus with hybrid retrieval & cross-encoder reranking..."
            if active_domain_key == "user_corpus"
            else "Analyzing sample legal corpus, executing hardware-filtered hybrid retrieval & cross-encoder reranking..."
        )
        with st.spinner(spinner_msg):
            result = query_backend(
                query=user_input,
                domain=active_domain_key,
                doc_filter=selected_doc_filter
            )

        answer_text = result.get("answer", "No answer generated.")
        st.markdown(answer_text)

        # Metadata badges
        active_domain = result.get("domain", "")
        latency = result.get("execution_time_seconds", 0.0)
        if active_domain == "user_corpus":
            dom_badge = "badge-user"
            dom_label = "Personal Uploaded Corpus"
        elif active_domain == "commercial_contracts":
            dom_badge = "badge-contracts"
            dom_label = "Commercial Contracts (Sample)"
        else:
            dom_badge = "badge-const"
            dom_label = "Constitutional Law (Sample)"

        st.markdown(
            f'<span class="badge-pill {dom_badge}">🏛️ {dom_label}</span>'
            f'<span class="badge-pill badge-latency">⏱️ {latency}s</span>',
            unsafe_allow_html=True
        )

        # Citations drawer
        citations = result.get("citations", [])
        if citations:
            with st.expander(f"📚 Verifiable Source Citations ({len(citations)} sources)", expanded=True):
                for cit in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">Rank #{cit['rank']}: {cit['section']}</div>
                        <div class="citation-meta">Document: {cit['document_name']} | Reference: {cit['citation_id']}</div>
                        <div class="citation-text">{cit['excerpt']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Persist to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": {
                "domain": active_domain,
                "latency": latency
            },
            "citations": citations
        })
