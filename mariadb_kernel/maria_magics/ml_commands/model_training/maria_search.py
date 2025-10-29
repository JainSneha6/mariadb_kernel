# maria_kernel/maria_magics/maria_search.py
"""
%maria_search

Hybrid BM25 + vector search. Hardcoded settings:
 - MODEL_NAME = "all-MiniLM-L6-v2"
 - K = 8
 - BM25_WEIGHT = 0.3
 - CANDIDATE_N = 500

Usage:
  %maria_search query="refund policy for returns"
  %maria_search "refund policy for returns"    # raw-line fallback
If no query supplied, defaults to "testquery".
"""

import shlex
import json
import logging
import re
import numpy as np
from distutils import util

# optional sentence-transformers
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# optional pandas for parsing HTML tables
_PANDAS_AVAILABLE = False
try:
    import pandas as _pd
    _PANDAS_AVAILABLE = True
except Exception:
    _PANDAS_AVAILABLE = False

try:
    from mariadb_kernel.maria_magics.maria_magic import MariaMagic
except Exception:
    # lightweight fallback if run standalone for tests
    class MariaMagic:
        def __init__(self, *a, **k):
            pass
        def type(self): return "Line"
        def name(self): return "maria_search"
        def help(self): return "Search (hybrid)."


class MariaSearch(MariaMagic):
    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger("MariaSearch")

        # Hardcoded (per your request)
        self.MODEL_NAME = "all-MiniLM-L6-v2"
        self.K = 8
        self.CANDIDATE_N = 500
        self.BM25_WEIGHT = 0.3

    def type(self):
        return "Line"

    def name(self):
        return "maria_search"

    def help(self):
        return "%maria_search query=\"text\" — hybrid BM25 + vector search (hardcoded model/weights)"

    # ----------------- utilities -----------------
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
        if input_str == "":
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

    def _parse_html_table(self, html):
        """Return a list-of-dicts or pandas.DataFrame. Best-effort fallback if pandas missing."""
        if html is None:
            return None
        if _PANDAS_AVAILABLE:
            try:
                dfs = _pd.read_html(html)
                if dfs:
                    return dfs[0]
            except Exception:
                pass
        # fallback simple parser -> list of dicts
        try:
            tbl = re.search(r"<table[^>]*>(.*?)</table>", str(html), flags=re.S | re.I)
            if not tbl:
                return None
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl.group(1), flags=re.S | re.I)
            if not rows:
                return None
            headers = None
            parsed = []
            for r in rows:
                # find header cells
                if headers is None:
                    ths = re.findall(r"<th[^>]*>(.*?)</th>", r, flags=re.S | re.I)
                    if ths:
                        headers = [re.sub(r"<[^>]+>", "", c).strip() for c in ths]
                        continue
                tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S | re.I)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in tds]
                if not cells:
                    continue
                if headers and len(cells) == len(headers):
                    parsed.append(dict(zip(headers, cells)))
                else:
                    parsed.append({str(i): cells[i] if i < len(cells) else "" for i in range(len(cells))})
            return parsed
        except Exception:
            return None

    def _is_nonempty_table(self, table):
        """Return True if 'table' (DataFrame or list-of-dicts) has at least one row."""
        if table is None:
            return False
        if _PANDAS_AVAILABLE and hasattr(_pd, "DataFrame") and isinstance(table, _pd.DataFrame):
            return not table.empty
        if isinstance(table, list):
            return len(table) > 0
        # other truthy checks (strings etc) considered empty for our use
        return False

    def _embed_texts(self, texts, dim=384):
        if len(texts) == 0:
            return np.zeros((0, dim), dtype=np.float32)
        if _ST_AVAILABLE:
            try:
                st = SentenceTransformer(self.MODEL_NAME)
                embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                embs = np.array(embs, dtype=np.float32)
                if embs.ndim == 1:
                    embs = np.expand_dims(embs, 0)
                if embs.shape[1] != dim:
                    self.log.warning("Embedding dim mismatch: model returned %d, expected %d. Adjusting.",
                                     embs.shape[1], dim)
                    if embs.shape[1] > dim:
                        embs = embs[:, :dim].astype(np.float32)
                    else:
                        pad = np.zeros((embs.shape[0], dim - embs.shape[1]), dtype=np.float32)
                        embs = np.concatenate([embs, pad], axis=1)
                return embs
            except Exception as e:
                self.log.exception("sentence-transformers failed, falling back to deterministic embeddings: %s", e)
        rng = np.random.RandomState(12345)
        embs = rng.normal(size=(len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs

    def _parse_vector_literal(self, val):
        """Parse JSON or text vector into numpy array."""
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                return np.array(val, dtype=np.float32)
            except Exception:
                pass
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                return np.array(parsed, dtype=np.float32)
            except Exception:
                pass
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
        if not nums:
            return None
        try:
            arr = np.array([float(x) for x in nums], dtype=np.float32)
            return arr
        except Exception:
            return None

    # ----------------- main -----------------
    def execute(self, kernel, data):
        # parse args & query
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
            query = "testquery"
        query = str(query).strip()
        if not query:
            kernel._send_message("stderr", "Empty query; nothing to search.\n")
            return

        kernel._send_message("stdout", f"[debug] running hybrid search for query (len={len(query)}): {query}\n")

        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            kernel._send_message("stderr", "No mariadb_client available on kernel (can't run search).\n")
            return

        # determine DB
        try:
            db_html = mariadb_client.run_statement("SELECT DATABASE();")
            dbname = None
            parsed_db = self._parse_html_table(db_html)
            if parsed_db is None:
                m = re.search(r"<td[^>]*>(.*?)</td>", str(db_html), flags=re.S)
                dbname = m.group(1).strip() if m else ""
            else:
                if _PANDAS_AVAILABLE and hasattr(parsed_db, "iloc") and isinstance(parsed_db, _pd.DataFrame):
                    if not parsed_db.empty:
                        dbname = str(parsed_db.iloc[0, 0])
                    else:
                        dbname = ""
                elif isinstance(parsed_db, list) and len(parsed_db) > 0:
                    first = parsed_db[0]
                    if isinstance(first, dict):
                        dbname = next(iter(first.values()))
                    else:
                        dbname = first.get("0") if "0" in first else ""
                else:
                    dbname = ""
        except Exception as e:
            kernel._send_message("stderr", f"Failed to query current database: {e}\n")
            return

        if not dbname:
            kernel._send_message("stderr", "No current database selected (use `USE <db>` before running the magic).\n")
            return

        # --- BM25 prefilter if requested ---
        candidates = []
        try:
            if self.BM25_WEIGHT > 0:
                q_esc = self._sql_escape(query)
                sql = (
                    f"SELECT id AS chunk_id, doc_id, chunk_index, chunk_text, "
                    f"MATCH(chunk_text) AGAINST ({q_esc} IN NATURAL LANGUAGE MODE) AS bm25_score "
                    f"FROM `{dbname}`.`chunks` "
                    f"WHERE MATCH(chunk_text) AGAINST ({q_esc} IN NATURAL LANGUAGE MODE) "
                    f"ORDER BY bm25_score DESC LIMIT {self.CANDIDATE_N};"
                )
                html = mariadb_client.run_statement(sql)
                df = self._parse_html_table(html)
                if self._is_nonempty_table(df):
                    if _PANDAS_AVAILABLE and hasattr(_pd, "DataFrame") and isinstance(df, _pd.DataFrame):
                        for _, row in df.iterrows():
                            try:
                                cid = int(row.get("chunk_id") if "chunk_id" in row else row.get("id"))
                            except Exception:
                                cid = None
                            candidates.append({
                                "chunk_id": cid,
                                "chunk_text": row.get("chunk_text") if "chunk_text" in row else "",
                                "doc_id": row.get("doc_id") if "doc_id" in row else "",
                                "bm25_score": float(row.get("bm25_score") if "bm25_score" in row else 0.0) if row is not None else 0.0
                            })
                    else:
                        # parsed list-of-dicts
                        for r in df:
                            try:
                                cid = int(r.get("chunk_id") or r.get("id") or next(iter(r.values())))
                            except Exception:
                                cid = None
                            candidates.append({
                                "chunk_id": cid,
                                "chunk_text": r.get("chunk_text") or r.get(next(iter(r.keys()))) or "",
                                "doc_id": r.get("doc_id") or "",
                                "bm25_score": float(r.get("bm25_score") or 0.0)
                            })
        except Exception as e:
            kernel._send_message("stderr", f"BM25 prefilter failed: {e}\n")

        # if no candidates from BM25, fallback to sample
        if not candidates:
            try:
                sql_sample = (
                    f"SELECT id AS chunk_id, doc_id, chunk_index, chunk_text "
                    f"FROM `{dbname}`.`chunks` "
                    f"ORDER BY RAND() LIMIT {self.CANDIDATE_N};"
                )
                html = mariadb_client.run_statement(sql_sample)
                df = self._parse_html_table(html)
                if self._is_nonempty_table(df):
                    if _PANDAS_AVAILABLE and hasattr(_pd, "DataFrame") and isinstance(df, _pd.DataFrame):
                        for _, row in df.iterrows():
                            try:
                                cid = int(row.get("chunk_id") if "chunk_id" in row else row.get("id"))
                            except Exception:
                                cid = None
                            candidates.append({
                                "chunk_id": cid,
                                "chunk_text": row.get("chunk_text") if "chunk_text" in row else "",
                                "doc_id": row.get("doc_id") if "doc_id" in row else ""
                            })
                    else:
                        for r in df:
                            try:
                                cid = int(r.get("chunk_id") or r.get("id") or next(iter(r.values())))
                            except Exception:
                                cid = None
                            candidates.append({
                                "chunk_id": cid,
                                "chunk_text": r.get("chunk_text") or "",
                                "doc_id": r.get("doc_id") or ""
                            })
            except Exception as e:
                kernel._send_message("stderr", f"Candidate sampling failed: {e}\n")

        if not candidates:
            kernel._send_message("stderr", "No candidate chunks found (empty chunks table?).\n")
            return

        candidate_ids = [int(c["chunk_id"]) for c in candidates if c.get("chunk_id") is not None]
        if not candidate_ids:
            kernel._send_message("stderr", "No valid candidate chunk ids.\n")
            return

        # --- fetch embeddings: try native embeddings table first ---
        id_list_sql = ",".join(str(int(x)) for x in candidate_ids)
        emb_rows = None
        try:
            sql_emb = (
                f"SELECT e.chunk_id, e.embedding_vector, c.chunk_text, c.doc_id, c.chunk_meta "
                f"FROM `{dbname}`.`embeddings` e "
                f"JOIN `{dbname}`.`chunks` c ON e.chunk_id = c.id "
                f"WHERE e.chunk_id IN ({id_list_sql});"
            )
            html = mariadb_client.run_statement(sql_emb)
            emb_rows = self._parse_html_table(html)
        except Exception:
            emb_rows = None

        emb_map = {}
        # parse native embeddings if returned
        if self._is_nonempty_table(emb_rows):
            if _PANDAS_AVAILABLE and hasattr(_pd, "DataFrame") and isinstance(emb_rows, _pd.DataFrame):
                for _, row in emb_rows.iterrows():
                    try:
                        cid = int(row.get("chunk_id") if "chunk_id" in row else row.get("chunk_id"))
                    except Exception:
                        continue
                    emb_raw = row.get("embedding_vector") if "embedding_vector" in row else row.get("embedding") if "embedding" in row else None
                    vec = self._parse_vector_literal(emb_raw)
                    if vec is None:
                        continue
                    norm = np.linalg.norm(vec)
                    if norm == 0: norm = 1.0
                    emb_map[cid] = {
                        "vec": vec.astype(np.float32) / norm,
                        "chunk_text": row.get("chunk_text") if "chunk_text" in row else "",
                        "doc_id": row.get("doc_id") if "doc_id" in row else "",
                        "chunk_meta": row.get("chunk_meta") if "chunk_meta" in row else ""
                    }
            else:
                for r in emb_rows:
                    try:
                        cid = int(r.get("chunk_id") or r.get(next(iter(r.keys()))))
                    except Exception:
                        continue
                    emb_raw = r.get("embedding_vector") or r.get("embedding_json") or r.get("embedding_bin") or None
                    vec = self._parse_vector_literal(emb_raw)
                    if vec is None:
                        continue
                    norm = np.linalg.norm(vec)
                    if norm == 0: norm = 1.0
                    emb_map[cid] = {
                        "vec": vec.astype(np.float32) / norm,
                        "chunk_text": r.get("chunk_text") or "",
                        "doc_id": r.get("doc_id") or "",
                        "chunk_meta": r.get("chunk_meta") or ""
                    }

        # If native embeddings empty for candidates, try embeddings_json fallback
        if not emb_map:
            try:
                sql_json = (
                    f"SELECT ej.chunk_id, ej.embedding_json, c.chunk_text, c.doc_id, c.chunk_meta "
                    f"FROM `{dbname}`.`embeddings_json` ej "
                    f"JOIN `{dbname}`.`chunks` c ON ej.chunk_id = c.id "
                    f"WHERE ej.chunk_id IN ({id_list_sql});"
                )
                html_json = mariadb_client.run_statement(sql_json)
                rows_json = self._parse_html_table(html_json)
                if self._is_nonempty_table(rows_json):
                    if _PANDAS_AVAILABLE and hasattr(_pd, "DataFrame") and isinstance(rows_json, _pd.DataFrame):
                        for _, row in rows_json.iterrows():
                            try:
                                cid = int(row.get("chunk_id") if "chunk_id" in row else row.get(0))
                            except Exception:
                                continue
                            emb_raw = row.get("embedding_json") if "embedding_json" in row else row.get("embedding") or None
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
                            norm = np.linalg.norm(vec)
                            if norm == 0: norm = 1.0
                            emb_map[cid] = {
                                "vec": vec.astype(np.float32) / norm,
                                "chunk_text": row.get("chunk_text") if "chunk_text" in row else "",
                                "doc_id": row.get("doc_id") if "doc_id" in row else "",
                                "chunk_meta": row.get("chunk_meta") if "chunk_meta" in row else ""
                            }
                    else:
                        for r in rows_json:
                            try:
                                cid = int(r.get("chunk_id") or next(iter(r.values())))
                            except Exception:
                                continue
                            emb_raw = r.get("embedding_json") or r.get(next(iter([k for k in r.keys() if 'embedding' in k.lower()])), None)
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
                            norm = np.linalg.norm(vec)
                            if norm == 0: norm = 1.0
                            emb_map[cid] = {
                                "vec": vec.astype(np.float32) / norm,
                                "chunk_text": r.get("chunk_text") or "",
                                "doc_id": r.get("doc_id") or "",
                                "chunk_meta": r.get("chunk_meta") or ""
                            }
            except Exception:
                pass

        if not emb_map:
            kernel._send_message("stderr", "No embeddings found for candidate chunks (neither native nor JSON fallback).\n")
            return

        # compute query embedding (dim inferred from first vector)
        try:
            vec_dim = next(iter(emb_map.values()))["vec"].shape[0]
        except Exception:
            kernel._send_message("stderr", "Failed to determine embedding dimensionality.\n")
            return

        try:
            q_emb = self._embed_texts([query], dim=vec_dim)[0]
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0: q_norm = 1.0
            q_emb = q_emb.astype(np.float32) / q_norm
        except Exception as e:
            kernel._send_message("stderr", f"Failed to compute query embedding: {e}\n")
            return

        # combine scores and rank
        results = []
        bm25_scores = [float(c.get("bm25_score", 0.0) or 0.0) for c in candidates]
        bm25_max = max(bm25_scores) if bm25_scores else 0.0
        for c in candidates:
            cid = c.get("chunk_id")
            if cid not in emb_map:
                continue
            emb_info = emb_map[cid]
            sim = float(np.dot(q_emb, emb_info["vec"]))
            bm25_raw = float(c.get("bm25_score", 0.0) or 0.0)
            bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else 0.0
            combined = (self.BM25_WEIGHT * bm25_norm) + ((1.0 - self.BM25_WEIGHT) * ((sim + 1.0) / 2.0))
            results.append({
                "chunk_id": cid,
                "chunk_text": emb_info.get("chunk_text") or c.get("chunk_text", ""),
                "doc_id": emb_info.get("doc_id") or c.get("doc_id", ""),
                "chunk_meta": emb_info.get("chunk_meta") or c.get("chunk_meta", ""),
                "vec_sim": sim,
                "bm25": bm25_raw,
                "score": combined
            })

        if not results:
            kernel._send_message("stderr", "No scored results to return after filtering.\n")
            return

        results.sort(key=lambda r: r["score"], reverse=True)
        topk = results[: self.K]

        # output table
        lines = []
        header = ["chunk_id", "chunk_text...", "score", "vec_sim", "bm25", "doc_id"]
        lines.append("\t".join(header))
        for r in topk:
            text_preview = (r["chunk_text"] or "").replace("\n", " ")
            if len(text_preview) > 200:
                text_preview = text_preview[:197] + "..."
            score_s = f"{r['score']:.6f}"
            vec_s = f"{r['vec_sim']:.6f}"
            bm25_s = f"{r['bm25']:.6f}"
            line = "\t".join([str(r["chunk_id"]), text_preview, score_s, vec_s, bm25_s, str(r.get("doc_id", ""))])
            lines.append(line)

        out = "\n".join(lines) + "\n"
        kernel._send_message("stdout", out)
        return
