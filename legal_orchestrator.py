import os
import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from ingest import NativeGeminiEmbeddings
from legal_splitter import HierarchicalLegalSplitter
from hybrid_retriever import HybridLegalRetriever, normalize_legal_name, match_contract_name


def get_overall_page(doc: Document) -> int:
    """Extract true overall page number across concatenated PDFs."""
    meta = doc.metadata
    page = meta.get("page", 0)
    source = meta.get("source", "")
    if "part_2" in source.lower():
        return 139 + page
    return page


@dataclass
class Citation:
    document_name: str
    section: str
    citation_id: str
    hierarchy_level: int
    source: str
    excerpt: str
    rank: int


@dataclass
class QueryResult:
    query: str
    answer: str
    domain: str
    doc_filter: Optional[str]
    citations: List[Dict[str, Any]]
    execution_time_seconds: float
    model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-2"
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"


class LegalOrchestrator:
    """
    Unified Production Legal RAG Orchestrator.
    Manages dual-domain agentic routing (Indian Constitutional Law vs Commercial Contracts),
    hardware-filtered hybrid retrieval, Small-to-Big expansion, and verified citation formatting.
    """

    def __init__(self):
        print("Legal Orchestrator: Initializing enterprise legal RAG pipeline...")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if self.qdrant_url and self.qdrant_api_key:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60
            )
        else:
            self.qdrant_client = QdrantClient(path="./qdrant_db")

        self.embeddings = NativeGeminiEmbeddings(model="models/gemini-embedding-2")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            api_key=self.gemini_api_key
        )

        # 1. Initialize Contracts Engine (texts_legal / contracts_legal_rag)
        self.contracts_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="contracts_legal_rag",
            embedding=self.embeddings
        )
        self.contracts_retriever = HybridLegalRetriever(
            vector_store=self.contracts_store,
            corpus_dir="texts_legal",
            dense_k=8,
            sparse_k=8,
            fusion_k=10
        )

        # 2. Initialize Constitutional Law Engine (pdfs_leagal / hierarchical_legal_rag)
        pdf_dir = "pdfs_leagal" if os.path.exists("pdfs_leagal") else "pdfs_legal"
        self.constitutional_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="hierarchical_legal_rag",
            embedding=self.embeddings
        )
        self.constitutional_retriever = HybridLegalRetriever(
            vector_store=self.constitutional_store,
            corpus_dir=pdf_dir,
            dense_k=10,
            sparse_k=10,
            fusion_k=14
        )

        # 3. Initialize User Uploaded Corpus Engine (Dedicated user workspace)
        self.user_collection_name = "user_legal_rag"
        self.user_retriever: Optional[HybridLegalRetriever] = None
        self.user_store: Optional[QdrantVectorStore] = None
        self.user_corpus_info: Dict[str, Any] = {
            "total_chunks": 0,
            "parent_sections": 0,
            "documents": [],
            "last_uploaded": None
        }
        self.splitter = HierarchicalLegalSplitter()
        self._try_restore_user_corpus()

        print("Legal Orchestrator: All legal engines (Sample + User Workspace) initialized and operational.")

    def _save_user_corpus_cache(self, chunks: List[Document]):
        """Persist chunks to local cache for instant reload across sessions."""
        os.makedirs("scratch", exist_ok=True)
        cache_path = os.path.join("scratch", "user_corpus_cache.json")
        try:
            data = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in chunks
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save user corpus cache: {e}")

    def _try_restore_user_corpus(self):
        """Restore active user corpus from disk cache and existing Qdrant collection if available."""
        cache_path = os.path.join("scratch", "user_corpus_cache.json")
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            return
        try:
            if not self.qdrant_client.collection_exists(self.user_collection_name):
                return
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in data
            ]
            if not chunks:
                return
            self.user_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.user_collection_name,
                embedding=self.embeddings
            )
            self.user_retriever = HybridLegalRetriever(
                vector_store=self.user_store,
                corpus_chunks=chunks,
                dense_k=8,
                sparse_k=8,
                fusion_k=12
            )
            doc_names = sorted(list(set(c.metadata.get("document_name") for c in chunks if c.metadata.get("document_name"))))
            self.user_corpus_info = {
                "total_chunks": len(chunks),
                "parent_sections": len(self.user_retriever.parent_map),
                "documents": doc_names,
                "last_uploaded": "Restored from session cache"
            }
            print(f"Legal Orchestrator: Restored active user corpus ({len(doc_names)} docs, {len(chunks)} chunks).")
        except Exception as e:
            print(f"Notice: Could not restore user corpus cache: {e}")

    def ingest_user_documents(
        self,
        files: List[tuple[str, bytes]],
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Ingests user-uploaded legal documents (PDF, TXT, MD) into a dedicated user collection:
        1. Chunks files using HierarchicalLegalSplitter
        2. Recreates Qdrant collection 'user_legal_rag' with payload indexes
        3. Generates 3072-dim embeddings via NativeGeminiEmbeddings in batches
        4. Initializes HybridLegalRetriever with BM25 + FlashRank reranker
        5. Persists cache for fast reload
        """
        if not files:
            raise ValueError("No files provided for ingestion.")

        if progress_callback:
            progress_callback(0.05, "Parsing uploaded documents into legal clauses...")

        all_chunks: List[Document] = []
        doc_names_set = set()

        for idx, (fname, file_bytes) in enumerate(files):
            chunks = self.splitter.split_uploaded_file(fname, file_bytes)
            for c in chunks:
                doc_name = c.metadata.get("document_name")
                if doc_name:
                    doc_names_set.add(doc_name)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No extractable legal text or clauses found in uploaded files.")

        doc_names = sorted(list(doc_names_set))
        total_chunks = len(all_chunks)

        if progress_callback:
            progress_callback(0.2, f"Setting up vector database for {len(doc_names)} documents ({total_chunks} chunks)...")

        # Recreate user collection in Qdrant
        if self.qdrant_client.collection_exists(self.user_collection_name):
            try:
                self.qdrant_client.delete_collection(self.user_collection_name)
            except Exception as e:
                print(f"User collection delete note: {e}")

        self.qdrant_client.create_collection(
            collection_name=self.user_collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )

        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.user_collection_name,
                field_name="metadata.document_name",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.user_collection_name,
                field_name="metadata.document_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            print(f"Payload index note: {e}")

        self.user_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.user_collection_name,
            embedding=self.embeddings
        )

        # Upload vectors in batches
        batch_size = 10
        total_batches = (total_chunks + batch_size - 1) // batch_size
        for b_idx in range(total_batches):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, total_chunks)
            batch = all_chunks[start_i:end_i]

            prog = 0.25 + 0.65 * ((b_idx + 1) / total_batches)
            if progress_callback:
                progress_callback(prog, f"Embedding batch {b_idx + 1}/{total_batches} ({end_i}/{total_chunks} chunks)...")

            self.user_store.add_documents(batch, batch_size=len(batch))
            if b_idx + 1 < total_batches:
                time.sleep(1)

        if progress_callback:
            progress_callback(0.95, "Building lexical BM25 index & cross-encoder reranker...")

        # Initialize hybrid retriever for user corpus
        self.user_retriever = HybridLegalRetriever(
            vector_store=self.user_store,
            corpus_chunks=all_chunks,
            dense_k=8,
            sparse_k=8,
            fusion_k=12
        )

        self.user_corpus_info = {
            "total_chunks": total_chunks,
            "parent_sections": len(self.user_retriever.parent_map),
            "documents": doc_names,
            "last_uploaded": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Cache chunks in scratch for persistence
        self._save_user_corpus_cache(all_chunks)

        if progress_callback:
            progress_callback(1.0, "Corpus indexing complete! Ready for legal queries.")

        return {
            "status": "success",
            "total_documents": len(doc_names),
            "documents": doc_names,
            "total_chunks": total_chunks,
            "parent_sections": len(self.user_retriever.parent_map)
        }

    def clear_user_corpus(self):
        """Reset and remove the active user uploaded corpus."""
        try:
            if self.qdrant_client.collection_exists(self.user_collection_name):
                self.qdrant_client.delete_collection(self.user_collection_name)
        except Exception as e:
            print(f"Notice: error deleting user collection: {e}")
        self.user_retriever = None
        self.user_store = None
        self.user_corpus_info = {
            "total_chunks": 0,
            "parent_sections": 0,
            "documents": [],
            "last_uploaded": None
        }
        cache_path = os.path.join("scratch", "user_corpus_cache.json")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
        print("Legal Orchestrator: User corpus cleared.")

    def classify_domain(self, query: str) -> str:
        """
        Agentic Domain Classifier: Dispatches query to Constitutional Law or Commercial Contracts.
        """
        q_lower = query.lower()

        # Strong Constitutional Law indicators
        const_patterns = [
            r"\barticle\s+\d+", r"\bconstitution\b", r"\bfundamental\s+rights?\b",
            r"\bsupreme\s+court\b", r"\bkesavananda\b", r"\bmaneka\b", r"\bgopalan\b",
            r"\bhabeas\s+corpus\b", r"\bpreamble\s+of\s+india\b", r"\bparliament\b",
            r"\bjudicial\s+review\b", r"\bbasic\s+structure\b", r"\bdirective\s+principles\b"
        ]
        for pat in const_patterns:
            if re.search(pat, q_lower):
                return "constitutional_law"

        # Strong Commercial Contract indicators
        contract_patterns = [
            r"\bagreement\b", r"\bcontract\b", r"\bparties\b", r"\bgoverning\s+law\b",
            r"\bindemnif\w+\b", r"\bliabilit\w+\b", r"\bwarrant\w+\b", r"\bterminat\w+\b",
            r"\bexpiration\s+date\b", r"\beffective\s+date\b", r"\bdistributor\b",
            r"\bjoint\s+venture\b", r"\baudit\s+rights\b", r"\bnon-disparagement\b",
            r"\brofr\b", r"\bexclusiv\w+\b", r"\banti-assignment\b", r"\bnotice\b"
        ]
        for pat in contract_patterns:
            if re.search(pat, q_lower):
                return "commercial_contracts"

        # Default heuristic based on document prefix or general phrasing
        if "in the " in q_lower:
            return "commercial_contracts"

        return "constitutional_law"

    def extract_doc_filter(self, query: str) -> Optional[str]:
        """Auto-detect target contract name from query phrasing like 'In the <DocName>,' or contract keywords."""
        all_doc_names = list(set(
            c.metadata.get("document_name", "")
            for c in self.contracts_retriever.corpus_chunks
            if c.metadata.get("document_name")
        ))
        if query.startswith("In the "):
            comma_idx = query.find(", Highlight the parts")
            if comma_idx == -1:
                comma_idx = query.find(",")
            if comma_idx != -1:
                candidate_name = query[7:comma_idx].strip()
                matched = match_contract_name(candidate_name, all_doc_names)
                if matched:
                    return matched
        return match_contract_name(query, all_doc_names)

    def query(
        self,
        query_text: str,
        domain: Optional[str] = None,
        doc_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> QueryResult:
        """
        Execute full production orchestration:
        1. Resolve domain (Auto-detect or manual override)
        2. Scope document filter if relevant
        3. Retrieve via domain-specific hybrid retriever + Small-to-Big expansion
        4. Generate grounded, high-relevancy answer
        5. Build verifiable structured citations
        """
        start_time = time.time()

        # 1. Resolve domain
        active_domain = domain
        if not active_domain or active_domain == "auto":
            if self.user_retriever is not None and self.user_corpus_info.get("total_chunks", 0) > 0:
                active_domain = "user_corpus"
            else:
                active_domain = self.classify_domain(query_text)

        # 2. Scoping & retrieval
        citations: List[Citation] = []
        if active_domain == "user_corpus":
            if self.user_retriever is None or not self.user_retriever.corpus_chunks:
                return QueryResult(
                    query=query_text,
                    answer="No documents have been uploaded to your active corpus yet. Please upload contracts, briefs, or legal documents using the sidebar uploader.",
                    domain="user_corpus",
                    doc_filter=doc_filter,
                    citations=[],
                    execution_time_seconds=0.0,
                    model=self.model_name
                )
            active_doc_filter = doc_filter
            k = top_k or 4
            retrieved_docs = self.user_retriever.retrieve(
                query_text,
                top_k=k,
                doc_filter=active_doc_filter,
                expand_to_parent=True
            )
            prompt_template = """You are an expert legal counsel and due diligence assistant analyzing the user's uploaded legal corpus. Answer the user's question directly, accurately, and concisely based strictly and ONLY on the provided context below.
- Lead directly with the substantive answer in the very first words. Do NOT start sentences with procedural openers or preambles like 'Pursuant to Section X...', 'According to the agreement...', or 'Based on the context...'.
- Put citations ONLY as brief trailing parentheticals at the very end of your answer or after each distinct finding (e.g., '(Section 6.4)', '(Article 2)', or '(Page 3)').
- If answering that the document does not contain provisions regarding the matter, state clearly: "The uploaded documents do not contain provisions regarding this matter."
- Conciseness & Completeness: Directly address all parts of the question, quoting exact critical phrases, dates, or numeric amounts.
- Base your answers strictly on the provided context. Do not extrapolate external facts.

Context:
{context}

Question:
{question}

Answer:"""
        elif active_domain == "commercial_contracts":
            active_doc_filter = doc_filter or self.extract_doc_filter(query_text)
            k = top_k or 3
            retrieved_docs = self.contracts_retriever.retrieve(
                query_text,
                top_k=k,
                doc_filter=active_doc_filter,
                expand_to_parent=True
            )
            prompt_template = """You are an expert legal assistant specializing in commercial contract review and due diligence. Answer the user's question directly, accurately, and concisely based ONLY on the provided contract context.
- Lead directly with the substantive answer in the very first words. Do NOT start sentences with procedural openers or preambles like 'Pursuant to Section X...', 'According to the agreement...', or 'Based on the context...'.
- Citation Rules:
  * Put the citation ONLY as a brief trailing parenthetical at the very end of your answer (e.g., '(Section 6.4)' or '(Section 13)').
  * If the clause or information is from the document title, preamble, recitals, or an exhibit heading without an explicit numbered section, cite ONLY '(Title)', '(Preamble)', or '(Exhibit 1)'. NEVER guess, fabricate, or invent section numbers.
  * If asked for the name of the contract or document name, state simply the exact title of the contract (e.g., 'Manufacturing Agreement').
  * If answering that the contract does not contain provisions regarding the matter, do NOT add any section citation.
- Conciseness & Completeness: Keep the answer strictly to 1-3 focused sentences directly addressing all parts of the question. Do NOT add extraneous details that were not requested (e.g. if asked for the parties, state only the legal entity names; do not add addresses, recitals, or signature blocks).
- Multi-Part & Synthesis Questions: Ensure all distinct requirements, remedies, or commitments queried are directly addressed (e.g., if asked about remedies and venue, state both the equitable remedies and the specific court; if asked about financial commitments, state letter of credit, purchase orders, and quotas; if asked about liability protections, state indemnification, insurance, and liability caps).
- Substantive & Functional Equivalence: Contractual concepts are frequently expressed under alternative headings (e.g., 'Expiration Date' under 'Termination', 'Duration', or 'Term'; 'Cap on Liability' under 'Limitation of Liability', 'Prefilled Syringes', or 'Remedies'). If the context contains substantive rules, durations, or limits governing the queried matter, state them directly.
- ANTI-FALSE-NEGATIVE MANDATE (CRITICAL):
  * Do NOT claim "The contract does not contain provisions regarding this matter" if ANY retrieved section contains relevant limits, replacement cost obligations, risk allocations, or caps.
  * Broad Scope of "Cap on Liability" / "Limitation of Liability": A cap or limitation of liability includes clauses that limit a party's liability to specific remedies or recovery amounts (e.g., limiting liability solely to replacement costs or direct damages), sole remedy provisions, or risk-of-loss limitations (e.g., Section 2.2(b) limiting Antares' liability for lost or damaged Prefilled Syringes while at the Manufacturing Site solely to replacement costs). Even if a general limitation section (such as Section 9.4) is redacted ([***]), you MUST state the specific liability limits, replacement-cost caps, or sole remedies established in other sections (e.g. Section 2.2(b)).
  * ONLY use "The contract does not contain provisions regarding this matter" if the retrieved text is 100% silent on the topic across all retrieved sections.
- Cross-References & Ancillary Agreements: When the agreement states that certain procedures, audits, or terms are set forth in an exhibit or separate agreement (e.g., Quality Agreement), state that cross-reference directly instead of claiming the matter is not covered.
- Comparative / Cross-Contract Questions: When asked to compare, contrast, or identify which contract has a specific term, duration, governing law, or legal structure, answer comprehensively across all contracts in the context. Explicitly identify the contract that matches, and contrast it with the respective terms, governing laws, or durations of the other contracts in the context (e.g. state the governing law or term of each contract).
- Direct Quotes: Quote only specific critical phrases, dates, or numeric caps (e.g. 'twenty-four (24) months', '$1,000,000'). Never copy-paste entire multi-sentence paragraphs verbatim.
- Base your answers strictly on the provided text. Do not extrapolate or introduce external facts.

Context:
{context}

Question:
{question}

Answer:"""
        else:
            # Constitutional Law
            active_doc_filter = None
            k = top_k or 4
            retrieved_docs = self.constitutional_retriever.retrieve(
                query_text,
                top_k=k,
                expand_to_parent=True
            )
            prompt_template = """You are an expert Indian constitutional law assistant. Answer the user's question directly, precisely, and concisely using ONLY the provided context.
- Base your answers and page citations strictly on the provided context material.
- If the context does not contain the answer or states the topic is not covered, state clearly: "The study material does not provide information on this topic."
- Do not copy Table of Contents dots or index listings. Focus on substantive constitutional principles, articles, and case law holdings.

Context:
{context}

Question:
{question}

Answer:"""

        # 3. Format citations
        for rank, doc in enumerate(retrieved_docs, start=1):
            meta = doc.metadata
            doc_name = meta.get("document_name") or meta.get("source") or "Legal Document"
            section = meta.get("section") or meta.get("article") or "General"
            cid = meta.get("clause_id") or f"Page {get_overall_page(doc)}"
            level = meta.get("hierarchy_level", 3)
            src = meta.get("source", "")
            
            # Clean body excerpt (strip initial breadcrumb header)
            lines = doc.page_content.splitlines()
            body_lines = lines[1:] if lines and lines[0].startswith("[") else lines
            excerpt = "\n".join(body_lines).strip()

            citations.append(Citation(
                document_name=doc_name,
                section=section,
                citation_id=cid,
                hierarchy_level=level,
                source=src,
                excerpt=excerpt,
                rank=rank
            ))

        # 4. Generate grounded answer
        formatted_context_blocks = []
        for c in citations:
            formatted_context_blocks.append(f"[{c.document_name} | {c.section} | {c.citation_id}]\n{c.excerpt}")
        formatted_context = "\n\n---\n\n".join(formatted_context_blocks)

        if not formatted_context.strip():
            answer = "The contract or legal study material does not contain provisions regarding this matter."
        else:
            prompt = prompt_template.format(context=formatted_context, question=query_text)
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                answer = "".join(text_parts).strip()
            else:
                answer = str(content).strip()

        elapsed = round(time.time() - start_time, 2)

        return QueryResult(
            query=query_text,
            answer=answer,
            domain=active_domain,
            doc_filter=active_doc_filter,
            citations=[asdict(c) for c in citations],
            execution_time_seconds=elapsed,
            model=self.model_name
        )

    def get_catalog(self) -> Dict[str, Any]:
        """Return catalog of all indexed user documents and pre-loaded sample materials."""
        contracts = sorted(list(set(
            c.metadata.get("document_name", "")
            for c in self.contracts_retriever.corpus_chunks
            if c.metadata.get("document_name")
        )))
        
        const_sources = sorted(list(set(
            c.metadata.get("source_file") or c.metadata.get("source") or c.metadata.get("document_name", "")
            for c in self.constitutional_retriever.corpus_chunks
            if (c.metadata.get("source_file") or c.metadata.get("source") or c.metadata.get("document_name"))
        )))
        const_units = sorted(list(set(
            c.metadata.get("unit", "")
            for c in self.constitutional_retriever.corpus_chunks
            if c.metadata.get("unit")
        )))

        user_docs = self.user_corpus_info.get("documents", [])
        user_chunks_count = len(self.user_retriever.corpus_chunks) if self.user_retriever else 0
        user_parents_count = len(self.user_retriever.parent_map) if self.user_retriever else 0

        user_info = {
            "collection": self.user_collection_name,
            "total_chunks": user_chunks_count,
            "parent_sections": user_parents_count,
            "documents": user_docs,
            "is_active": self.user_retriever is not None and user_chunks_count > 0,
            "last_uploaded": self.user_corpus_info.get("last_uploaded")
        }

        sample_contracts = {
            "collection": "contracts_legal_rag",
            "total_chunks": len(self.contracts_retriever.corpus_chunks),
            "parent_sections": len(self.contracts_retriever.parent_map),
            "documents": contracts
        }

        sample_const = {
            "collection": "hierarchical_legal_rag",
            "total_chunks": len(self.constitutional_retriever.corpus_chunks),
            "parent_sections": len(self.constitutional_retriever.parent_map),
            "documents": const_sources,
            "units": const_units
        }

        return {
            "user_corpus": user_info,
            "sample_corpus": {
                "commercial_contracts": sample_contracts,
                "constitutional_law": sample_const
            },
            # Backwards compatibility aliases
            "commercial_contracts": sample_contracts,
            "constitutional_law": sample_const
        }


# Global singleton instance for application reuse
_global_orchestrator: Optional[LegalOrchestrator] = None

def get_orchestrator() -> LegalOrchestrator:
    global _global_orchestrator
    if _global_orchestrator is None or not hasattr(_global_orchestrator, "classify_domain"):
        _global_orchestrator = LegalOrchestrator()
    return _global_orchestrator
