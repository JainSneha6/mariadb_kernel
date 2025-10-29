# mariadb_kernel/maria_magics/maria_rag_query.py
"""
%maria_rag_query

Single-command RAG: retrieve relevant chunks, run fusion chain (LLM via Gemini) and return answer.

Hardcoded settings:
 - retriever = "hybrid"
 - k = 6
 - llm_model = "gemini-2.5-flash"  (hardcoded)
 - prompt = "default"
 - bm25_weight = 0.3

Usage:
  %maria_rag_query query="How do I cancel my subscription?"
  %maria_rag_query query="How do I cancel my subscription?" explain=true

Notes:
 - The code will attempt to use the Google GenAI Python client (google.genai). It checks
   the environment variables GOOGLE_API_KEY or GENAI_API_KEY for the API key.
 - If the GenAI client or API key is unavailable, the magic falls back to a local fusion chain.
"""

import shlex
import json
import logging
import re
import os
import numpy as np
from distutils import util

from mariadb_kernel.maria_magics.maria_magic import MariaMagic

# optional sentence-transformers
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# optional Google GenAI (Gemini) client
_GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

class MariaRAGQuery(MariaMagic):
    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger("MariaRAGQuery")

        # HARDCODED SETTINGS (per your request)
        self.RETRIEVER = "hybrid"
        self.K = 6
        self.BM25_WEIGHT = 0.3
        self.CANDIDATE_N = 500
        self.LLM_MODEL = "gemini-2.5-flash"   # using Gemini model per your snippet
        self.PROMPT_NAME = "default"
        self.EMBED_DIM = 384

    def type(self):
        return "Line"

    def name(self):
        return "maria_rag_query"

    def help(self):
        return "%maria_rag_query query=\"...\" — retrieve+fusion RAG (hardcoded settings)"

    # ---------------- Parsing helpers ----------------
    def _str_to_obj(self, s):
        try:
            return int(s)
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            pass
        try:
            return bool(util.strtobool(s))
        except Exception:
            pass
        try:
            return json.loads(s)
        except Exception:
            pass
        if isinstance(s, str) and len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    def parse_args(self, input_obj):
        if input_obj is None:
            return {}
        if isinstance(input_obj, dict):
            return input_obj
        if not isinstance(input_obj, str):
            try:
                return dict(input_obj)
            except Exception:
                return {}
        input_str = input_obj.strip()
        if not input_str:
            return {}
        try:
            pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        except Exception:
            pairs = {}
            for token in input_str.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    pairs[k] = v
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _sql_escape(self, s):
        if s is None:
            return "NULL"
        if not isinstance(s, str):
            return str(s)
        return "'" + s.replace("'", "''") + "'"

    # ---------------- Embedding utilities ----------------
    def _embed_texts(self, texts, dim=None):
        """Return normalized numpy embeddings for texts. If sentence-transformers available use it."""
        if dim is None:
            dim = self.EMBED_DIM
        if len(texts) == 0:
            return np.zeros((0, dim), dtype=np.float32)
        if _ST_AVAILABLE:
            try:
                st = SentenceTransformer("all-MiniLM-L6-v2")
                embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                embs = np.array(embs, dtype=np.float32)
                if embs.ndim == 1:
                    embs = np.expand_dims(embs, 0)
                if embs.shape[1] != dim:
                    if embs.shape[1] > dim:
                        embs = embs[:, :dim].astype(np.float32)
                    else:
                        pad = np.zeros((embs.shape[0], dim - embs.shape[1]), dtype=np.float32)
                        embs = np.concatenate([embs, pad], axis=1)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (embs / norms).astype(np.float32)
            except Exception as e:
                self.log.debug("sentence-transformers failure: %s", e)
        # deterministic fallback
        rng = np.random.RandomState(12345)
        embs = rng.normal(size=(len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embs / norms).astype(np.float32)

    def _parse_html_table(self, html):
        """Best-effort HTML -> list-of-dicts parser used for mariadb_client outputs."""
        if html is None:
            return None
        s = str(html)
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S | re.I)
        parsed = []
        header = []
        for r in rows:
            ths = re.findall(r"<th[^>]*>(.*?)</th>", r, flags=re.S | re.I)
            if ths and not header:
                header = [re.sub(r"<[^>]+>", "", t).strip() for t in ths]
                continue
            tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S | re.I)
            if not tds:
                continue
            cells = [re.sub(r"<[^>]+>", "", t).strip() for t in tds]
            if header and len(cells) == len(header):
                parsed.append(dict(zip(header, cells)))
            else:
                parsed.append({str(i): cells[i] if i < len(cells) else "" for i in range(len(cells))})
        return parsed if parsed else None

    def _parse_vector_literal(self, val):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.array(val, dtype=np.float32)
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return np.array(arr, dtype=np.float32)
            except Exception:
                pass
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
        if not nums:
            return None
        try:
            return np.array([float(x) for x in nums], dtype=np.float32)
        except Exception:
            return None

    # ---------------- Retrieval helpers ----------------
    def _bm25_prefilter(self, kernel, dbname, query_text):
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            return []
        q_esc = self._sql_escape(query_text)
        sql = (
            f"SELECT id AS chunk_id, doc_id, chunk_index, chunk_text, "
            f"MATCH(chunk_text) AGAINST ({q_esc} IN NATURAL LANGUAGE MODE) AS bm25_score "
            f"FROM `{dbname}`.`chunks` "
            f"WHERE MATCH(chunk_text) AGAINST ({q_esc} IN NATURAL LANGUAGE MODE) "
            f"ORDER BY bm25_score DESC LIMIT {self.CANDIDATE_N};"
        )
        try:
            html = mariadb_client.run_statement(sql)
            rows = self._parse_html_table(html)
            if not rows:
                return []
            cand = []
            for r in rows:
                try:
                    cid = int(r.get("chunk_id") or r.get("id"))
                except Exception:
                    continue
                cand.append({
                    "chunk_id": cid,
                    "doc_id": r.get("doc_id") or "",
                    "chunk_index": int(r.get("chunk_index") or r.get("0") or 0),
                    "chunk_text": r.get("chunk_text") or "",
                    "bm25_score": float(r.get("bm25_score") or 0.0)
                })
            return cand
        except Exception as e:
            self.log.debug("BM25 query failed: %s", e)
            return []

    def _sample_candidates(self, kernel, dbname):
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            return []
        sql = f"SELECT id AS chunk_id, doc_id, chunk_index, chunk_text FROM `{dbname}`.`chunks` ORDER BY RAND() LIMIT {self.CANDIDATE_N};"
        try:
            html = mariadb_client.run_statement(sql)
            rows = self._parse_html_table(html)
            if not rows:
                return []
            cand = []
            for r in rows:
                try:
                    cid = int(r.get("chunk_id") or r.get("id"))
                except Exception:
                    continue
                cand.append({
                    "chunk_id": cid,
                    "doc_id": r.get("doc_id") or "",
                    "chunk_index": int(r.get("chunk_index") or 0),
                    "chunk_text": r.get("chunk_text") or ""
                })
            return cand
        except Exception as e:
            self.log.debug("Sampling failed: %s", e)
            return []

    def _fetch_embeddings_for_candidates(self, kernel, dbname, candidate_ids):
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            return {}
        if not candidate_ids:
            return {}

        ids_sql = ",".join(str(int(x)) for x in candidate_ids)
        # first attempt native embeddings join
        try:
            sql = (
                f"SELECT e.chunk_id, e.embedding_vector, c.chunk_text, c.doc_id, c.chunk_index "
                f"FROM `{dbname}`.`embeddings` e "
                f"JOIN `{dbname}`.`chunks` c ON e.chunk_id = c.id "
                f"WHERE e.chunk_id IN ({ids_sql});"
            )
            html = mariadb_client.run_statement(sql)
            rows = self._parse_html_table(html)
            emb_map = {}
            if rows:
                for r in rows:
                    try:
                        cid = int(r.get("chunk_id") or r.get("0"))
                    except Exception:
                        continue
                    emb_raw = r.get("embedding_vector") or r.get("embedding") or None
                    vec = self._parse_vector_literal(emb_raw)
                    if vec is None:
                        continue
                    norm = np.linalg.norm(vec) or 1.0
                    emb_map[cid] = {
                        "vec": (vec / norm).astype(np.float32),
                        "chunk_text": r.get("chunk_text") or "",
                        "doc_id": r.get("doc_id") or "",
                        "chunk_index": int(r.get("chunk_index") or 0)
                    }
            if emb_map:
                return emb_map
        except Exception as e:
            self.log.debug("native embeddings fetch failed: %s", e)

        # fallback to embeddings_json
        try:
            sql_json = (
                f"SELECT ej.chunk_id, ej.embedding_json, c.chunk_text, c.doc_id, c.chunk_index "
                f"FROM `{dbname}`.`embeddings_json` ej "
                f"JOIN `{dbname}`.`chunks` c ON ej.chunk_id = c.id "
                f"WHERE ej.chunk_id IN ({ids_sql});"
            )
            html_json = mariadb_client.run_statement(sql_json)
            rows_json = self._parse_html_table(html_json)
            emb_map = {}
            if rows_json:
                for r in rows_json:
                    try:
                        cid = int(r.get("chunk_id") or r.get("0"))
                    except Exception:
                        continue
                    emb_raw = r.get("embedding_json") or r.get("embedding") or None
                    vec = None
                    if emb_raw is not None:
                        try:
                            if isinstance(emb_raw, (list, tuple)):
                                vec = np.array(emb_raw, dtype=np.float32)
                            else:
                                vec = np.array(json.loads(emb_raw), dtype=np.float32)
                        except Exception:
                            vec = self._parse_vector_literal(emb_raw)
                    if vec is None:
                        continue
                    norm = np.linalg.norm(vec) or 1.0
                    emb_map[cid] = {
                        "vec": (vec / norm).astype(np.float32),
                        "chunk_text": r.get("chunk_text") or "",
                        "doc_id": r.get("doc_id") or "",
                        "chunk_index": int(r.get("chunk_index") or 0)
                    }
            return emb_map
        except Exception as e:
            self.log.debug("embeddings_json fetch failed: %s", e)
            return {}

    # ---------------- Gemini LLM call ----------------
    def _call_gemini(self, system_prompt, user_prompt, model_name=None, max_output_tokens=400):
        """
        Call Gemini using google.genai per the snippet the user provided.
        Looks for API key in GOOGLE_API_KEY or GENAI_API_KEY environment variables.
        Returns (text, raw_response) or (None, None) on failure.
        """
        if not _GENAI_AVAILABLE:
            self.log.debug("google.genai not available in environment.")
            return None, None

        api_key = "AIzaSyBW1n6kIu0o-W3l0-pMBOMc4nzjYfsbETg"
        if not api_key:
            self.log.debug("No GENAI API key found in GOOGLE_API_KEY or GENAI_API_KEY.")
            return None, None

        try:
            client = genai.Client(api_key=api_key)
            # Build combined content: put system + user into 'contents' - simple approach
            contents = system_prompt + "\n\n" + user_prompt
            resp = client.models.generate_content(
                model=model_name or self.LLM_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            # The user's snippet used resp.text
            text = getattr(resp, "text", None)
            if text is None:
                # some genai client versions put result in resp.output or resp.candidates
                try:
                    # try attribute "candidates"
                    if hasattr(resp, "candidates") and resp.candidates:
                        text = getattr(resp.candidates[0], "content", None) or getattr(resp.candidates[0], "text", None)
                    elif hasattr(resp, "output"):
                        text = str(resp.output)
                    else:
                        text = str(resp)
                except Exception:
                    text = str(resp)
            return text, resp
        except Exception as e:
            self.log.debug("Gemini call failed: %s", e)
            return None, None

    # ---------------- Local fusion fallback ----------------
    def _fusion_chain_local(self, question, context_blocks):
        """
        Local, deterministic fusion map-reduce:
        - Map: pick sentences from each context containing question tokens
        - Reduce: join deduplicated sentences into a compact answer
        """
        q = question.lower()
        q_tokens = set(re.findall(r"\w+", q))
        picked = []
        evidence = []
        debug = {"map": [], "reduce": None}

        for b in context_blocks:
            text = b["chunk_text"]
            sentences = re.split(r'(?<=[\.\?\!])\s+', text)
            picks = []
            for s in sentences:
                st = s.strip()
                if not st:
                    continue
                s_tokens = set(re.findall(r"\w+", st.lower()))
                if len(q_tokens & s_tokens) > 0:
                    picks.append(st)
            if not picks and sentences:
                picks = [sentences[0].strip()]
            picks = picks[:3]
            if picks:
                picked.extend(picks)
                evidence.append({
                    "doc_id": b["doc_id"],
                    "chunk_index": b["chunk_index"],
                    "snippet": " ".join(picks)[:400]
                })
            debug["map"].append({"chunk_id": b["chunk_id"], "picked_count": len(picks)})

        # Reduce: deduplicate and join
        uniq = []
        seen = set()
        for s in picked:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                uniq.append(s.strip())

        if not uniq:
            answer = "I couldn't find a clear answer in the retrieved documents."
        else:
            answer = " ".join(uniq[:8])
            if len(answer) > 800:
                answer = answer[:797] + "..."
        debug["reduce"] = {"picked_sentences": len(uniq)}
        return answer, evidence, debug

    # ---------------- Main entry ----------------
    def execute(self, kernel, data):
        # parse args and query
        try:
            args = self.parse_args(self.args)
        except Exception as e:
            kernel._send_message("stderr", f"Error parsing arguments: {e}\n")
            args = {}

        query = None
        if isinstance(args, dict):
            query = args.get("query") or args.get("q")
        if not query:
            if isinstance(data, str) and data.strip():
                query = data.strip()
        if not query:
            kernel._send_message("stderr", "No query supplied. Usage: %maria_rag_query query=\"...\"\n")
            return

        explain = False
        if isinstance(args, dict):
            if args.get("explain") in (True, "true", "True", 1, "1"):
                explain = True

        kernel._send_message("stdout", f"[debug] RAG query received (len={len(query)}): {query}\n")
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            kernel._send_message("stderr", "No mariadb_client available on kernel (can't run retrieval).\n")
            return

        # determine DB
        try:
            db_html = mariadb_client.run_statement("SELECT DATABASE();")
            db_parsed = self._parse_html_table(db_html)
            dbname = ""
            if db_parsed and isinstance(db_parsed, list) and len(db_parsed) > 0:
                first = db_parsed[0]
                dbname = next(iter(first.values()))
            else:
                m = re.search(r"<td[^>]*>(.*?)</td>", str(db_html), flags=re.S)
                if m:
                    dbname = m.group(1).strip()
        except Exception as e:
            kernel._send_message("stderr", f"Failed to detect current DB: {e}\n")
            return

        if not dbname:
            kernel._send_message("stderr", "No current database selected (use `USE <db>` before running the magic).\n")
            return

        # RETRIEVAL: BM25 prefilter (hybrid)
        candidates = []
        if self.RETRIEVER == "hybrid":
            candidates = self._bm25_prefilter(kernel, dbname, query)
        if not candidates:
            candidates = self._sample_candidates(kernel, dbname)
        if not candidates:
            kernel._send_message("stderr", "No candidate chunks found (chunks table empty?).\n")
            return

        candidate_ids = [c["chunk_id"] for c in candidates if c.get("chunk_id") is not None]
        emb_map = self._fetch_embeddings_for_candidates(kernel, dbname, candidate_ids)
        if not emb_map:
            kernel._send_message("stderr", "No embeddings found for any candidate chunks.\n")
            return

        # compute query embedding consistent with vector dim
        first_vec = next(iter(emb_map.values()))["vec"]
        vec_dim = first_vec.shape[0]
        q_emb = self._embed_texts([query], dim=vec_dim)[0]

        # combine bm25 + vector
        scored = []
        bm25_values = [float(c.get("bm25_score", 0.0) or 0.0) for c in candidates]
        bm25_max = max(bm25_values) if bm25_values else 0.0
        for c in candidates:
            cid = c.get("chunk_id")
            if cid not in emb_map:
                continue
            emb_info = emb_map[cid]
            sim = float(np.dot(q_emb, emb_info["vec"]))
            bm25_raw = float(c.get("bm25_score", 0.0) or 0.0)
            bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else 0.0
            combined = (self.BM25_WEIGHT * bm25_norm) + ((1.0 - self.BM25_WEIGHT) * ((sim + 1.0) / 2.0))
            scored.append({
                "chunk_id": cid,
                "doc_id": emb_info.get("doc_id"),
                "chunk_index": emb_info.get("chunk_index"),
                "chunk_text": emb_info.get("chunk_text"),
                "vec_sim": sim,
                "bm25": bm25_raw,
                "score": combined
            })

        if not scored:
            kernel._send_message("stderr", "No scored candidates after combining BM25/vector.\n")
            return

        # top-K
        scored.sort(key=lambda r: r["score"], reverse=True)
        topk = scored[: self.K]

        # assemble context blocks with citations (internal only)
        context_blocks = []
        for s in topk:
            context_blocks.append({
                "chunk_id": s["chunk_id"],
                "doc_id": s["doc_id"],
                "chunk_index": s["chunk_index"],
                "chunk_text": s["chunk_text"],
                "vec_sim": s["vec_sim"],
                "bm25": s["bm25"],
                "score": s["score"]
            })

        # Build prompt / context to send to Gemini
        context_text = ""
        for i, b in enumerate(context_blocks):
            citation = f"[{b['doc_id']}::chunk_{b['chunk_index']}]"
            context_text += f"--- SOURCE {i+1} {citation} ---\n{b['chunk_text']}\n\n"

        system_prompt = "You are a helpful assistant that answers questions based on provided documents. When you use information from a source include a citation tag like [DOCID::chunk_X]."
        user_prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context_text}\n\nINSTRUCTIONS:\nAnswer the question concisely.\n"

        # Try Gemini via google.genai
        llm_answer = None
        llm_raw_resp = None
        gemini_text, gemini_raw = self._call_gemini(system_prompt, user_prompt, model_name=self.LLM_MODEL, max_output_tokens=512)
        if gemini_text:
            llm_answer = gemini_text
            llm_raw_resp = gemini_raw

        chain_debug = None
        if not llm_answer:
            ans, evidence, debug = self._fusion_chain_local(query, context_blocks)
            chain_debug = debug
            # **DO NOT** append sources to the answer (per request)
            llm_answer = ans

        # Output answer only (no sources printed)
        kernel._send_message("stdout", "\n=== ANSWER ===\n")
        kernel._send_message("stdout", llm_answer + "\n\n")

        # NOTE: sources are intentionally NOT printed here.

        if explain:
            kernel._send_message("stdout", "\n=== EXPLAIN: retrieval candidates (top 20 shown) ===\n")
            for s in scored[:20]:
                kernel._send_message("stdout", f"chunk_id={s['chunk_id']} doc_id={s['doc_id']} chunk_index={s['chunk_index']} score={s['score']:.6f} vec_sim={s['vec_sim']:.6f} bm25={s['bm25']:.6f}\n")
            if chain_debug is not None:
                kernel._send_message("stdout", "\n=== EXPLAIN: chain debug ===\n")
                kernel._send_message("stdout", json.dumps(chain_debug, indent=2) + "\n")
            if llm_raw_resp is not None:
                kernel._send_message("stdout", "\n=== GEMINI RAW RESP (truncated) ===\n")
                kernel._send_message("stdout", str(llm_raw_resp)[:2000] + "\n")

        return
