# mariadb_kernel/maria_magics/maria_ingest.py
import shlex
import json
import math
import logging
import numpy as np
from distutils import util
from mariadb_kernel.maria_magics.maria_magic import MariaMagic
from mariadb_kernel.mariadb_client import MariaDBClient

# optional sentence-transformers
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# IPython history fallback
try:
    from IPython import get_ipython
except Exception:
    get_ipython = None


class MariaIngest(MariaMagic):
    """
    Cell magic to ingest text documents into MariaDB, chunk them, and store embeddings
    directly in a native VECTOR(384) column.

    Usage (cell magic):
    %%maria_ingest doc_id=DOC1 title="My Doc" chunk_size=800 overlap=100 text="My document text here"
    <optional cell body will be ignored if text=... is provided>

    The magic still supports passing the document body in the cell. If `text` is present
    in the magic args it will be used in preference to the cell body.
    """
    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger("MariaIngest")

    def type(self):
        return "Cell"

    def name(self):
        return "maria_ingest"

    def help(self):
        return "Ingest a document: chunk -> store -> embeddings (model fixed to all-MiniLM-L6-v2). Accepts `text=...` to pass the document content in the magic args."

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
        """
        Accept either:
         - dict (already parsed)
         - string (key=value tokens)
        Returns a dict of parsed args.
        """
        if input_obj is None:
            return {}
        # already a dict (kernel may pass args as dict)
        if isinstance(input_obj, dict):
            return input_obj
        # if it's not a str, try to convert
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
        except Exception as e:
            # fallback: naive split on spaces for simple cases
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

    def _simple_chunk(self, text: str, chunk_size: int, overlap: int):
        if not text:
            return []
        t = text.strip()
        if len(t) <= chunk_size:
            return [t]

        chunks = []
        start = 0
        L = len(t)
        while start < L:
            end = min(L, start + chunk_size)
            if end < L:
                look_ahead = t[end: min(L, end + 100)]
                idx_nl = look_ahead.find("\n")
                idx_dot = look_ahead.find(".")
                if idx_nl != -1:
                    end += idx_nl + 1
                elif idx_dot != -1:
                    end += idx_dot + 1
            chunk = t[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= L:
                break
            start = max(0, end - overlap)

        if not chunks and t:
            chunks = [t]
        return chunks

    def _embed_batch(self, texts, dim=384):
        model_name = "all-MiniLM-L6-v2"
        if len(texts) == 0:
            return np.zeros((0, dim), dtype=np.float32)

        if _ST_AVAILABLE:
            try:
                st = SentenceTransformer(model_name)
                embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                embs = np.array(embs, dtype=np.float32)
                if embs.ndim == 1:
                    embs = np.expand_dims(embs, 0)
                if embs.shape[1] != dim:
                    self.log.warning("Embedding dim mismatch: model returned %d, expected %d. Adjusting.", embs.shape[1], dim)
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

    def _parse_single_result(self, html):
        if html is None:
            return None
        try:
            import pandas as _pd
            df = _pd.read_html(html)[0]
            if df.size == 0:
                return None
            return df.iloc[0, 0]
        except Exception:
            try:
                import re
                m = re.search(r"<td[^>]*>(.*?)</td>", html, flags=re.S)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
        return None

    def _get_cell_text_fallback(self):
        # fallback to IPython history if needed
        try:
            if get_ipython is None:
                return ""
            ip = get_ipython()
            if not ip:
                return ""
            ns = ip.user_ns
            if ns and "In" in ns:
                hist = ns["In"]
                if isinstance(hist, (list, tuple)) and len(hist) > 0:
                    for v in reversed(hist):
                        if isinstance(v, str) and v.strip():
                            return v
            hm = getattr(ip, "history_manager", None)
            if hm:
                try:
                    entries = list(hm.get_tail(1, include_latest=True))
                    if entries:
                        src = entries[-1][2]
                        if isinstance(src, str) and src.strip():
                            return src
                except Exception:
                    pass
        except Exception:
            pass
        return ""

    def execute(self, kernel, data):
        """Main entry point called by the MariaDB Jupyter kernel."""

        # --- Extract cell content robustly ---
        cell_text = ""
        try:
            # Case 1: standard MariaDB kernel -> {"cell": {"args": ..., "body": ...}}
            if isinstance(data, dict):
                if "cell" in data and isinstance(data["cell"], dict):
                    if "body" in data["cell"] and isinstance(data["cell"]["body"], str):
                        cell_text = data["cell"]["body"]
                    elif "code" in data["cell"] and isinstance(data["cell"]["code"], str):
                        cell_text = data["cell"]["code"]
                # Case 2: other kernels
                elif any(k in data for k in ("code", "content", "message", "data")):
                    for k in ("code", "content", "message", "data"):
                        if k in data and isinstance(data[k], str):
                            cell_text = data[k]
                            break
            elif isinstance(data, str):
                cell_text = data
            else:
                try:
                    cell_text = str(data)
                except Exception:
                    cell_text = ""
        except Exception as e:
            kernel._send_message("stderr", f"[debug] could not extract cell text: {e}")
            cell_text = ""

        if cell_text:
            cell_text = cell_text.strip()

        preview = cell_text[:80].replace("\n", " ") + ("..." if len(cell_text) > 80 else "")
        kernel._send_message("stdout", f"[debug] stored content length={len(cell_text)} preview={preview}\n")


        # --- Parse arguments (key=value pairs) ---
        try:
            args = self.parse_args(self.args)
        except Exception as e:
            kernel._send_message("stderr", f"Error parsing arguments: {e}")
            return

        # If the user provided a `text` argument, prefer it over the cell body.
        provided_text = args.get("text") if isinstance(args, dict) else None
        if isinstance(provided_text, str) and provided_text.strip():
            cell_text = provided_text
            kernel._send_message("stdout", f"[debug] using text from args (len={len(cell_text)})\n")

        # metadata
        doc_id = args.get("doc_id") or f"doc_{int(np.floor(np.random.random()*1e9))}"
        title = args.get("title") or ""
        chunk_size = int(args.get("chunk_size", 800) or 800)
        overlap = int(args.get("overlap", 100) or 100)
        dim = 384
        metadata = args.get("metadata", {}) or {}

        # build docs list
        docs_to_ingest = []
        maybe_json = (cell_text or "").strip()
        try:
            if maybe_json.startswith("[") or maybe_json.startswith("{"):
                parsed = json.loads(maybe_json)
                if isinstance(parsed, list):
                    for d in parsed:
                        docs_to_ingest.append({
                            "doc_id": d.get("doc_id") or d.get("id") or f"doc_{int(np.floor(np.random.random()*1e9))}",
                            "title": d.get("title") or "",
                            "content": d.get("content") or "",
                            "metadata": d.get("metadata") or {}
                        })
                elif isinstance(parsed, dict) and ("content" in parsed or "doc_id" in parsed):
                    docs_to_ingest.append({
                        "doc_id": parsed.get("doc_id") or parsed.get("id") or doc_id,
                        "title": parsed.get("title") or title,
                        "content": parsed.get("content") or "",
                        "metadata": parsed.get("metadata") or metadata
                    })
                else:
                    docs_to_ingest.append({
                        "doc_id": doc_id,
                        "title": title,
                        "content": cell_text,
                        "metadata": metadata
                    })
            else:
                docs_to_ingest.append({
                    "doc_id": doc_id,
                    "title": title,
                    "content": cell_text,
                    "metadata": metadata
                })
        except Exception:
            docs_to_ingest.append({
                "doc_id": doc_id,
                "title": title,
                "content": cell_text,
                "metadata": metadata
            })

        # db client
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            kernel._send_message("stderr", "No mariadb_client available on kernel (can't run ingestion).")
            return

        # determine current DB
        try:
            db_name_html = mariadb_client.run_statement("SELECT DATABASE();")
            dbname = self._parse_single_result(db_name_html) or ""
        except Exception as e:
            kernel._send_message("stderr", f"Failed to query current database: {e}")
            return

        if not dbname:
            kernel._send_message("stderr", "No current database selected (use `USE <db>` before running the magic).")
            return

        # create tables with VECTOR(384)
        try:
            mariadb_client.run_statement(
                f"""
                CREATE TABLE IF NOT EXISTS `{dbname}`.`documents` (
                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                  doc_id VARCHAR(191) UNIQUE,
                  title TEXT,
                  content LONGTEXT,
                  metadata JSON,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """
            )

            mariadb_client.run_statement(
                f"""
                CREATE TABLE IF NOT EXISTS `{dbname}`.`chunks` (
                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                  doc_id VARCHAR(191),
                  chunk_index INT,
                  chunk_text LONGTEXT,
                  chunk_meta JSON,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE KEY uq_doc_chunk (doc_id, chunk_index),
                  FULLTEXT KEY ft_chunk_text (chunk_text)
                ) ENGINE=InnoDB;
                """
            )

            # native VECTOR column
            mariadb_client.run_statement(
                f"""
                CREATE TABLE IF NOT EXISTS `{dbname}`.`embeddings` (
                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                  chunk_id BIGINT UNIQUE,
                  model VARCHAR(128),
                  dim INT,
                  embedding_vector VECTOR({dim}),
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """
            )

            # best-effort ANN index
            try:
                mariadb_client.run_statement(
                    f"CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON `{dbname}`.`embeddings` (embedding_vector) USING ANN;"
                )
            except Exception:
                try:
                    mariadb_client.run_statement(
                        f"CREATE INDEX idx_embeddings_vector ON `{dbname}`.`embeddings` (embedding_vector) USING ANN WITH (distance='cosine');"
                    )
                except Exception as e_idx:
                    self.log.debug("ANN index creation skipped/failed (ok): %s", e_idx)

        except Exception as e:
            kernel._send_message("stderr", f"DDL failed: {e}")
            return

        # ingest loop
        total_chunks = 0
        total_emb_rows = 0
        for doc in docs_to_ingest:
            d_doc_id = doc.get("doc_id")
            d_title = doc.get("title")
            d_content = doc.get("content") or ""
            d_meta = doc.get("metadata") or {}

            # insert document row
            try:
                mariadb_client.run_statement(
                    f"""
                    INSERT INTO `{dbname}`.`documents` (doc_id, title, content, metadata)
                    VALUES ({self._sql_escape(d_doc_id)}, {self._sql_escape(d_title)}, {self._sql_escape(d_content)}, {self._sql_escape(json.dumps(d_meta))})
                    ON DUPLICATE KEY UPDATE title=VALUES(title), content=VALUES(content), metadata=VALUES(metadata);
                    """
                )
            except Exception as e:
                kernel._send_message("stderr", f"Failed to insert document {d_doc_id}: {e}")
                continue

            # debug: fetch stored content
            try:
                res_html = mariadb_client.run_statement(
                    f"SELECT content FROM `{dbname}`.`documents` WHERE doc_id = {self._sql_escape(d_doc_id)} LIMIT 1;"
                )
                stored_content = self._parse_single_result(res_html) or ""
                kernel._send_message("stdout", f"[debug] stored content length={len(stored_content)}")
                if d_content and not stored_content:
                    kernel._send_message("stderr", "[warning] document content inserted into DB appears empty (possible client/encoding issue).")
            except Exception as e:
                kernel._send_message("stderr", f"Warning: could not verify stored document content: {e}")

            # chunk
            chunks = self._simple_chunk(d_content, chunk_size, overlap)
            if not chunks and d_content:
                chunks = [d_content]
            total_chunks += len(chunks)

            # insert chunks and collect ids
            inserted_chunk_ids = []
            for idx, chunk_text in enumerate(chunks):
                try:
                    mariadb_client.run_statement(
                        f"""
                        INSERT INTO `{dbname}`.`chunks` (doc_id, chunk_index, chunk_text, chunk_meta)
                        VALUES ({self._sql_escape(d_doc_id)}, {idx}, {self._sql_escape(chunk_text)}, {self._sql_escape(json.dumps({}))});
                        """
                    )
                    # LAST_INSERT_ID
                    try:
                        last_html = mariadb_client.run_statement("SELECT LAST_INSERT_ID();")
                        last_val = self._parse_single_result(last_html)
                        last_id = int(last_val) if last_val is not None else None
                    except Exception:
                        last_id = None

                    if last_id is not None:
                        inserted_chunk_ids.append((idx, last_id))
                    else:
                        # fallback lookup
                        try:
                            sel_html = mariadb_client.run_statement(
                                f"SELECT id FROM `{dbname}`.`chunks` WHERE doc_id = {self._sql_escape(d_doc_id)} AND chunk_index = {idx} LIMIT 1;"
                            )
                            sel_val = self._parse_single_result(sel_html)
                            inserted_chunk_ids.append((idx, int(sel_val)) if sel_val is not None else (idx, None))
                        except Exception:
                            inserted_chunk_ids.append((idx, None))
                except Exception as e:
                    kernel._send_message("stderr", f"Failed to insert chunk {idx} for {d_doc_id}: {e}")
                    inserted_chunk_ids.append((idx, None))
                    continue

            # diagnostics if nothing inserted
            if len(inserted_chunk_ids) == 0 and chunks:
                kernel._send_message("stderr", f"[debug] no chunk ids collected for doc {d_doc_id}; attempting to read existing chunks for diagnostic.")
                try:
                    full_sel = mariadb_client.run_statement(
                        f"SELECT id, chunk_index FROM `{dbname}`.`chunks` WHERE doc_id = {self._sql_escape(d_doc_id)} ORDER BY chunk_index;"
                    )
                    # best-effort parse
                    import pandas as _pd
                    try:
                        df = _pd.read_html(full_sel)[0]
                        tmp_map = {int(r["chunk_index"]): int(r["id"]) for _, r in df.iterrows()}
                        inserted_chunk_ids = [(i, tmp_map.get(i)) for i in range(len(chunks))]
                    except Exception:
                        pass
                except Exception:
                    pass

            # embeddings
            if chunks:
                embs = self._embed_batch(chunks, dim)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs_norm = (embs / norms).astype(np.float32)

                for (i, chunk_db_id), vec in zip(inserted_chunk_ids, embs_norm):
                    if chunk_db_id is None:
                        self.log.debug("No db chunk id for doc %s chunk %d — skipping embedding store", d_doc_id, i)
                        kernel._send_message("stderr", f"[debug] no chunk id for doc {d_doc_id} chunk {i}; embedding skipped.")
                        continue
                    vec_list = [float(v) for v in vec.tolist()]
                    vec_literal = "[" + ",".join(repr(x) for x in vec_list) + "]"
                    try:
                        mariadb_client.run_statement(
                            f"""
                            INSERT INTO `{dbname}`.`embeddings` (chunk_id, model, dim, embedding_vector)
                            VALUES ({chunk_db_id}, {self._sql_escape('all-MiniLM-L6-v2')}, {dim}, {vec_literal})
                            ON DUPLICATE KEY UPDATE model=VALUES(model), dim=VALUES(dim), embedding_vector=VALUES(embedding_vector);
                            """
                        )
                        total_emb_rows += 1
                    except Exception as e:
                        kernel._send_message("stderr", f"Failed to insert embedding for chunk_id={chunk_db_id}: {e}")
                        continue

        # final
        kernel._send_message("stdout", f"Ingest complete. documents={len(docs_to_ingest)} chunks_total={total_chunks} embeddings_written={total_emb_rows}\n")
        if total_chunks == 0:
            kernel._send_message("stderr", "Warning: no chunks were created. If your document text is present in the `documents` table but chunk_text is missing, check client encoding and ensure the cell body was passed to the kernel. Use `SELECT content FROM documents WHERE doc_id=\"...\";` to inspect.")
        kernel._send_message("stdout", "Notes:\n - embedding model used: all-MiniLM-L6-v2 (dim=384)\n - Native VECTOR column was created/used where available.\n")
        return
