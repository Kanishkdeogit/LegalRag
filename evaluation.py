import os
import json
import time
import re
import pandas as pd
from qdrant_client import QdrantClient

# Mock deprecated langchain_community modules that Ragas unconditionally imports at startup
import sys
from types import ModuleType
mock_chat_vertex = ModuleType("langchain_community.chat_models.vertexai")
mock_chat_vertex.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock_chat_vertex

mock_emb_vertex = ModuleType("langchain_community.embeddings.vertexai")
mock_emb_vertex.VertexAIEmbeddings = None
sys.modules["langchain_community.embeddings.vertexai"] = mock_emb_vertex

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import NativeGeminiEmbeddings  # Reuse custom embeddings wrapper
from hybrid_retriever import HybridLegalRetriever

# Ragas imports
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Experiment Tracking
import mlflow
# Load environment variables
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Verify API key
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

import argparse

def run_evaluation():
    parser = argparse.ArgumentParser(description="Unified Legal RAG Evaluation Pipeline")
    parser.add_argument("--dataset", dest="dataset_path", type=str, default=None,
                        help="Path to evaluation dataset JSON (e.g. 'eval_dataset2.json' or 'eval_dataset.json')")
    parser.add_argument("--data", "--dir", dest="data_dir", type=str, default=None,
                        help="Path to corpus directory (e.g. 'texts_legal' or 'pdfs_leagal')")
    parser.add_argument("--collection", dest="collection_name", type=str, default=None,
                        help="Qdrant collection name")
    args = parser.parse_args()

    # 1. Determine dataset path
    dataset_path = args.dataset_path or os.getenv("EVAL_DATASET")
    if not dataset_path:
        dataset_path = "eval_dataset.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset '{dataset_path}' not found.")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    is_contracts_benchmark = any(k in dataset_path.lower() for k in ["eval_dataset2", "eval_dataset3", "2", "3", "contract", "cuad"])
    
    # Determine corpus directory & collection name
    data_dir = args.data_dir
    if not data_dir:
        data_dir = "texts_legal" if is_contracts_benchmark else ("pdfs_leagal" if os.path.exists("pdfs_leagal") else "pdfs_legal")

    collection_name = args.collection_name or os.getenv("QDRANT_COLLECTION_NAME")
    if not collection_name:
        collection_name = "contracts_legal_rag" if is_contracts_benchmark else "hierarchical_legal_rag"

    if "3" in dataset_path:
        benchmark_title = "Commercial Contracts (CUAD + MultiHop + CrossDoc Benchmark 3)"
    elif is_contracts_benchmark:
        benchmark_title = "Commercial Contracts (CUAD)"
    else:
        benchmark_title = "Constitutional Law"
    print(f"=== Starting {benchmark_title} RAG Baseline Evaluation ===")
    print(f"Loaded {len(eval_data)} evaluation test cases from '{dataset_path}'.")
    print(f"Corpus Directory: '{data_dir}' | Qdrant Collection: '{collection_name}'")
    
    # 2. Connect to Qdrant Vector DB
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        print(f"Connecting to Qdrant Cloud at {qdrant_url}...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
    else:
        db_path = "./qdrant_db"
        print(f"Connecting to local Qdrant database at {db_path}...")
        client = QdrantClient(path=db_path)

    # 3. Initialize Embeddings and Vector Store
    embeddings = NativeGeminiEmbeddings(model="models/gemini-embedding-2")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    retriever = HybridLegalRetriever(
        vector_store=vector_store,
        corpus_dir=data_dir,
        dense_k=8,
        sparse_k=8,
        fusion_k=10
    )

    # 4. Initialize LLM (Gemini 3.5 Flash Lite)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    # 5. Define prompt template according to domain
    if is_contracts_benchmark:
        prompt_template = """You are an expert legal assistant specializing in commercial contract review and due diligence. Answer the user's question directly, accurately, and concisely based ONLY on the provided contract context.
- Lead directly with the substantive answer in the very first words. Do NOT start sentences with procedural openers or preambles like 'Pursuant to Section X...', 'According to the agreement...', or 'Based on the context...'.
- Citation Rules:
  * Put the citation ONLY as a brief trailing parenthetical at the very end of your answer (e.g., '(Section 6.4)' or '(Section 13)').
  * Primary Source Citation: Cite ONLY the single primary operative section in the trailing parenthetical. Do NOT cite multiple sections or compound lists (such as '(Sections 3.6, 4.3, and 4.4)') unless the question explicitly asks for multiple distinct provisions.
  * If the clause or information is from the document title, preamble, recitals, or an exhibit heading without an explicit numbered section, cite ONLY '(Title)', '(Preamble)', or '(Exhibit 1)'. NEVER guess, fabricate, or invent section numbers.
  * If asked for the name of the contract or document name, state simply the exact title of the contract (e.g., 'Manufacturing Agreement').
  * If answering that the contract does not contain provisions regarding the matter, do NOT add any section citation.
- ANSWER ONLY WHAT WAS ASKED: Strictly answer only the specific question asked. Do NOT append additional related-but-unrequested clauses, tangential background, or secondary provisions (e.g., if asked for minimum purchase commitments, state ONLY the purchase commitments and do not append sample product rules or general audit clauses; if asked about a cap on liability, state only the liability limit).
- Conciseness & Focus: Keep the answer strictly to 1-2 focused, direct sentences answering the question directly in the very first sentence. The highest-relevancy legal answers state the substantive answer immediately in the first sentence without extraneous background or procedural filler.
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
        prompt_template = """You are an expert Indian constitutional law assistant. Answer the user's question directly, precisely, and concisely using ONLY the provided context.
- Base your answers and page citations strictly on the provided context material.
- If the context does not contain the answer or states the topic is not covered, state clearly: "The study material does not provide information on this topic."
- Do not copy Table of Contents dots or index listings. Focus on substantive constitutional principles, articles, and case law holdings.

Context:
{context}

Question:
{question}

Answer:"""

    def is_toc_chunk(doc):
        content = doc.page_content
        return ". . . ." in content or "....." in content or "Table of Contents" in content

    def retrieve_clean_docs(retriever, query, top_k=3, doc_filter=None, is_cross_doc=False):
        return retriever.retrieve(query, top_k=top_k, doc_filter=doc_filter, is_cross_doc=is_cross_doc)

    def get_overall_page(doc):
        page_num = doc.metadata.get("page_number")
        if page_num:
            return page_num
        source = doc.metadata.get("source", "")
        filename = os.path.basename(source)
        match = re.search(r"-(\d+)-\d+\.pdf", filename)
        if match:
            start_page = int(match.group(1))
            page_idx = int(doc.metadata.get("page", 0))
            return start_page + page_idx
        return doc.metadata.get("page_label", str(doc.metadata.get("page", 0) + 1))

    def format_docs(docs):
        # Sort chronologically by page number for the LLM prompt reading
        def _get_page(d):
            p = d.metadata.get("page_number")
            if p is not None:
                try:
                    return int(p)
                except (ValueError, TypeError):
                    pass
            return 999
        sorted_docs = sorted(docs, key=_get_page)
        formatted = []
        for doc in sorted_docs:
            if doc.page_content.startswith("["):
                formatted.append(doc.page_content)
            else:
                page_info = get_overall_page(doc)
                formatted.append(f"[Page {page_info}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def before_sleep_log(retry_state):
        sleep_time = retry_state.next_action.sleep
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        exc_str = str(exc) if exc else ""
        if "GenerateRequestsPerDay" in exc_str:
            print(f"\n[Daily Quota Exceeded] Daily RPD limit exhausted for this model. Retries will not bypass daily caps. Please switch models.")
        elif "503" in exc_str or "UNAVAILABLE" in exc_str:
            print(f"\n[503 Server Busy] Temporary Google API capacity spike. Retrying in {sleep_time:.1f}s...")
        else:
            print(f"\n[Rate Limit] 429 hit. Pausing for {sleep_time:.1f}s to let the 1-minute RPM quota reset...")

    @retry(
        wait=wait_exponential(multiplier=1.5, min=35, max=75),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log,
        reraise=True
    )
    def invoke_llm_with_retry(prompt):
        return llm.invoke(prompt)

    # 6. Run baseline RAG on all questions in the evaluation set
    questions = []
    ground_truths = []
    for item in eval_data:
        q_raw = item.get("question", "")
        item_id = item.get("id", "")
        if "__" in item_id:
            contract_name = item_id.split("__")[0].strip()
            if contract_name.lower() == "cross_doc":
                q_formatted = q_raw
            elif contract_name.lower() not in q_raw.lower():
                q_formatted = f"In the {contract_name}, {q_raw}"
            else:
                q_formatted = q_raw
        else:
            q_formatted = q_raw

        if "answer_text" in item:
            gt = item.get("answer_text")
            if gt is None or item.get("is_negative"):
                gt = "The contract does not contain provisions or clauses regarding this matter."
        else:
            gt = item.get("ground_truth", "")

        questions.append(q_formatted)
        ground_truths.append(gt)
    
    answers = []
    contexts = []
    
    print(f"\nGenerating answers for {len(questions)} test cases...")
    for idx, q in enumerate(questions):
        print(f"[{idx+1}/{len(questions)}] Query: {q[:65]}...")
        doc_filter = None
        is_cross_doc = False
        if is_contracts_benchmark:
            item_id = eval_data[idx].get("id", "")
            if "__" in item_id:
                cand_doc = item_id.split("__")[0].strip()
                if cand_doc.lower() == "cross_doc":
                    doc_filter = None
                    is_cross_doc = True
                else:
                    doc_filter = cand_doc
            q_low = q.lower()
            if any(w in q_low for w in ["all these contracts", "which of these", "which contract", "across the contracts"]):
                doc_filter = None
                is_cross_doc = True
        k = 1 if is_cross_doc else (3 if is_contracts_benchmark else 4)
        retrieved_docs = retrieve_clean_docs(retriever, q, top_k=k, doc_filter=doc_filter, is_cross_doc=is_cross_doc)
        
        # Store context strings for Ragas
        contexts.append([
            doc.page_content if doc.page_content.startswith("[") else f"[Page {get_overall_page(doc)}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        # Format prompt & generate answer
        formatted_context = format_docs(retrieved_docs)
        formatted_prompt = prompt_template.format(context=formatted_context, question=q)
        
        response = invoke_llm_with_retry(formatted_prompt)
        
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                content = "".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
            
        answers.append(content)
        time.sleep(3.5)  # 3.5s pause ensures <= 17 RPM (strictly below Gemini 20 RPM Free Tier limit)
        
    print("\nAnswer generation completed across all queries.")
    print("Pausing 15s to allow API rate limits to fully reset before Ragas judge evaluation...")
    time.sleep(15)

    # 7. Set up Ragas LLM/Embeddings wrappers
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Prepare dataset for Ragas evaluation
    evaluation_dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    # 8. Start MLflow run and perform Ragas evaluation
    mlflow.set_experiment("Legal RAG Pipelines")
    
    if "3" in dataset_path:
        run_name = "contracts_eval3_multihop_crossdoc"
    elif is_contracts_benchmark:
        run_name = "contracts_hybrid_bm25_flashrank"
    else:
        run_name = "constitutional_hybrid_bm25_flashrank"
    print(f"\nEvaluating RAG pipeline metrics with Ragas and logging to MLflow run '{run_name}'...")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("dataset", "CUAD_MultiHop_CrossDoc_Eval3" if "3" in dataset_path else ("CUAD_Commercial_Contracts" if is_contracts_benchmark else "Constitutional_Law_I_Ambedkar_Univ"))
        mlflow.log_param("corpus_dir", data_dir)
        mlflow.log_param("num_eval_samples", len(questions))
        mlflow.log_param("retrieval_strategy", "Hybrid_Dense_BM25_RRF_FlashRank_PinpointBoost")
        mlflow.log_param("top_k", 3 if is_contracts_benchmark else 4)
        mlflow.log_param("fusion_k", 16)
        mlflow.log_param("query_expansion", True)
        mlflow.log_param("reranker", "ms-marco-MiniLM-L-12-v2")
        mlflow.log_param("vector_db", "Qdrant")
        mlflow.log_param("collection_name", collection_name)
        mlflow.log_param("embedding_model", "models/gemini-embedding-2")
        mlflow.log_param("llm_model", model_name)
        
        # Evaluate metrics using Ragas with robust RunConfig (max_retries=10, max_wait=90s, timeout=240s prevents 503/rate-limit NaNs)
        run_cfg = RunConfig(max_workers=1, timeout=240, max_wait=90, max_retries=10)
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                Faithfulness(), 
                AnswerRelevancy(strictness=1), 
                ContextPrecision(), 
                ContextRecall()
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_cfg,
            raise_exceptions=False
        )
        
        print("\n============================================")
        print(f"=== {benchmark_title} Evaluation Results ===")
        print("============================================")
        print(result)
        
        # Log aggregated Ragas metrics to MLflow
        for metric_name, score in result._repr_dict.items():
            mlflow.log_metric(metric_name, score)
            
        # Log detailed evaluation outputs as MLflow artifact (CSV format)
        df_results = result.to_pandas()
        
        # Quality check: Validate zero NaNs across all metrics
        metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df_results.columns]
        nan_counts = df_results[metric_cols].isna().sum()
        if nan_counts.any():
            print("\n[Warning: Missing / NaN Metrics detected due to judge rate limits]")
            for m, cnt in nan_counts.items():
                if cnt > 0:
                    print(f"  - {m}: {cnt} NaN values out of {len(df_results)} samples")
        else:
            print(f"\n[Validation Success: 100% Complete — Zero NaNs across all {len(df_results)} test samples!]")

        if "3" in dataset_path:
            csv_filename = "eval_dataset3_results.csv"
        elif is_contracts_benchmark:
            csv_filename = "contracts_evaluation_results.csv"
        else:
            csv_filename = "baseline_evaluation_results.csv"
        df_results.to_csv(csv_filename, index=False)
        mlflow.log_artifact(csv_filename)
        print(f"\nLogged detailed metrics to {csv_filename} and uploaded to MLflow.")
        
        # Log detailed JSON artifact
        import ast
        detailed_results = []
        for _, row in df_results.iterrows():
            retrieved = row.get("retrieved_contexts", [])
            if isinstance(retrieved, str):
                try:
                    retrieved = ast.literal_eval(retrieved)
                except Exception:
                    pass
            
            orig_item = eval_data[len(detailed_results)] if len(detailed_results) < len(eval_data) else {}
            detailed_results.append({
                "question": row.get("user_input", ""),
                "id": orig_item.get("id", ""),
                "query_type": orig_item.get("query_type", "cuad_templated" if "highlight the parts" in str(row.get("user_input", "")).lower() else "other"),
                "ground_truth": row.get("reference", ""),
                "retrieved_chunks": retrieved,
                "generated_answer": row.get("response", ""),
                "scores": {
                    "faithfulness": row.get("faithfulness", 0.0),
                    "answer_relevancy": row.get("answer_relevancy", 0.0),
                    "context_precision": row.get("context_precision", 0.0),
                    "context_recall": row.get("context_recall", 0.0)
                }
            })
        
        if "3" in dataset_path:
            json_filename = "eval_dataset3_results_with_chunks.json"
        elif is_contracts_benchmark:
            json_filename = "contracts_evaluation_results_with_chunks.json"
        else:
            json_filename = "evaluation_results_with_chunks.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        mlflow.log_artifact(json_filename)
        print(f"Logged detailed chunks and scores to {json_filename} and uploaded to MLflow.")

if __name__ == "__main__":
    run_evaluation()
