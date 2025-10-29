# mariadb_kernel/maria_magics/maria_ingest.py
import shlex
import json
import math
import logging
import os
import io
import re
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

# optional file extractors
_PYPDF2_AVAILABLE = False
_PYDOCX_AVAILABLE = False
try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except Exception:
    _PYPDF2_AVAILABLE = False
try:
    import docx
    _PYDOCX_AVAILABLE = True
except Exception:
    _PYDOCX_AVAILABLE = False

# IPython history fallback
try:
    from IPython import get_ipython
except Exception:
    get_ipython = None


class MariaIngest(MariaMagic):
    """
    Ingest text documents into MariaDB, chunk them, and store embeddings.

    This variant reduces noisy logging and prints only important status/warnings.
    """
    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger("MariaIngest")

    def type(self):
        return "Cell"

    def name(self):
        return "maria_ingest"

    def help(self):
        return "Ingest docs -> chunk -> embeddings. Uses native VECTOR when compatible; otherwise falls back to JSON. (cleaned logs)"

    # ---- utilities ----
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
                m = re.search(r"<td[^>]*>(.*?)</td>", html, flags=re.S)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
        return None

    def _read_file_content(self, path: str):
        warnings = []
        if not path:
            return "", warnings
        try:
            p = os.path.expanduser(os.path.expandvars(path))
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            if not os.path.exists(p):
                warnings.append(f"file not found: {p}")
                return "", warnings
            _, ext = os.path.splitext(p.lower())
            if ext in ('.txt', '.md', '.text', '.json', '.ndjson'):
                with io.open(p, 'r', encoding='utf-8', errors='replace') as fh:
                    return fh.read(), warnings
            if ext == '.pdf':
                if _PYPDF2_AVAILABLE:
                    try:
                        text_parts = []
                        with open(p, 'rb') as fh:
                            reader = PyPDF2.PdfReader(fh)
                            for page in reader.pages:
                                try:
                                    text_parts.append(page.extract_text() or '')
                                except Exception:
                                    pass
                        return ''.join(text_parts), warnings
                    except Exception as e:
                        warnings.append(f"PyPDF2 failed to extract PDF text: {e}")
                else:
                    warnings.append("PyPDF2 not available; cannot extract PDF text.")
                    return "", warnings
            if ext in ('.docx',):
                if _PYDOCX_AVAILABLE:
                    try:
                        doc = docx.Document(p)
                        paragraphs = [pr.text for pr in doc.paragraphs]
                        return '\n'.join(paragraphs), warnings
                    except Exception as e:
                        warnings.append(f"python-docx failed to extract docx: {e}")
                else:
                    warnings.append("python-docx not available; cannot extract docx text.")
                    return "", warnings
            try:
                with io.open(p, 'r', encoding='utf-8', errors='replace') as fh:
                    return fh.read(), warnings
            except Exception:
                try:
                    with io.open(p, 'r', encoding='latin-1', errors='replace') as fh:
                        return fh.read(), warnings
                except Exception as e:
                    warnings.append(f"Failed to read file: {e}")
                    return "", warnings
        except Exception as e:
            return "", [str(e)]

    def _get_existing_vector_dim(self, mariadb_client, dbname):
        """
        If an embeddings table exists, parse SHOW CREATE TABLE to extract the VECTOR(...) dimension.
        Returns int dimension if found, otherwise None.
        """
        try:
            resp = mariadb_client.run_statement("SHOW CREATE TABLE embeddings;")
            if not resp:
                return None
            txt = str(resp)
            m = re.search(r"embedding_vector\s+vector\((\d+)\)", txt, flags=re.I)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            m2 = re.search(r"vector\((\d+)\)", txt, flags=re.I)
            if m2:
                try:
                    return int(m2.group(1))
                except Exception:
                    return None
        except Exception:
            pass
        return None

    # ---- main execution ----
    def execute(self, kernel, data):
        # collect user-facing warnings/errors to print concisely at the end
        user_warnings = []

        # --- Extract cell content robustly ---
        cell_text = ""
        try:
            if isinstance(data, dict):
                if "cell" in data and isinstance(data["cell"], dict):
                    if "body" in data["cell"] and isinstance(data["cell"]["body"], str):
                        cell_text = data["cell"]["body"]
                    elif "code" in data["cell"] and isinstance(data["cell"]["code"], str):
                        cell_text = data["cell"]["code"]
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
            kernel._send_message("stderr", f"[debug] could not extract cell text: {e}\n")
            cell_text = ""

        cell_text = cell_text.strip() if cell_text else ""

        # --- Parse arguments ---
        try:
            args = self.parse_args(self.args)
        except Exception as e:
            kernel._send_message("stderr", f"Error parsing arguments: {e}\n")
            return

        # text arg or file arg preference
        provided_text = args.get("text") if isinstance(args, dict) else None
        file_arg = None
        for k in ("text_file", "file", "path"):
            if isinstance(args, dict) and args.get(k):
                file_arg = args.get(k)
                break

        if isinstance(provided_text, str) and provided_text.strip():
            cell_text = provided_text
            kernel._send_message("stdout", f"Using text from args (len={len(cell_text)})\n")
        elif file_arg:
            file_contents, warnings = self._read_file_content(file_arg)
            for w in warnings:
                user_warnings.append(w)
            if file_contents:
                cell_text = file_contents
                kernel._send_message("stdout", f"Using file content from {file_arg} (len={len(cell_text)})\n")
            else:
                kernel._send_message("stderr", f"Failed to read file or file contained no text: {file_arg}\n")

        # metadata and settings
        doc_id = args.get("doc_id") or f"doc_{int(np.floor(np.random.random()*1e9))}"
        title = args.get("title") or ""
        chunk_size = int(args.get("chunk_size", 800) or 800)
        overlap = int(args.get("overlap", 100) or 100)
        embedding_dim = 384  # expected embedding dim from model
        metadata = args.get("metadata", {}) or {}

        # build docs
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

        docs_to_ingest = [d for d in docs_to_ingest if (d.get("content") or "").strip()]
        if not docs_to_ingest:
            kernel._send_message("stderr", "No non-empty documents to ingest; aborting.\n")
            return

        # get mariadb client
        mariadb_client = getattr(kernel, "mariadb_client", None)
        if mariadb_client is None:
            kernel._send_message("stderr", "No mariadb_client available on kernel (can't run ingestion).\n")
            return

        # determine db
        try:
            db_name_html = mariadb_client.run_statement("SELECT DATABASE();")
            dbname = self._parse_single_result(db_name_html) or ""
            kernel._send_message("stdout", f"Using database: {dbname}\n")
        except Exception as e:
            kernel._send_message("stderr", f"Failed to query current database: {e}\n")
            return

        if not dbname:
            kernel._send_message("stderr", "No current database selected (use `USE <db>` before running the magic).\n")
            return

        # create tables: documents, chunks; embeddings handled carefully
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
            # create embeddings table if missing — keep existing definition if present
            try:
                mariadb_client.run_statement(
                    f"""
                    CREATE TABLE IF NOT EXISTS `{dbname}`.`embeddings` (
                      id BIGINT AUTO_INCREMENT PRIMARY KEY,
                      chunk_id BIGINT UNIQUE,
                      model VARCHAR(128),
                      dim INT,
                      embedding_vector VECTOR({embedding_dim}),
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB;
                    """
                )
            except Exception:
                # tolerate create failure and continue (we'll detect existing table schema)
                pass
        except Exception as e:
            kernel._send_message("stderr", f"DDL failed: {e}\n")
            return

        # detect existing VECTOR dimension (if any)
        existing_vec_dim = self._get_existing_vector_dim(mariadb_client, dbname)
        use_native_vector = True
        if existing_vec_dim is None:
            use_native_vector = True
        else:
            if existing_vec_dim != embedding_dim:
                use_native_vector = False
                user_warnings.append(f"embeddings.embedding_vector exists with dim={existing_vec_dim}; ingest dim={embedding_dim}. Will use JSON fallback.")

        # ensure embeddings_json exists (fallback)
        try:
            mariadb_client.run_statement(
                f"""
                CREATE TABLE IF NOT EXISTS `{dbname}`.`embeddings_json` (
                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                  chunk_id BIGINT UNIQUE,
                  model VARCHAR(128),
                  dim INT,
                  embedding_json JSON,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """
            )
        except Exception:
            pass

        # ingest loop with concise counters
        total_chunks = 0
        total_emb_rows = 0
        native_attempts = 0
        native_successes = 0
        native_failures = 0
        fallback_successes = 0
        fallback_failures = 0

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
                user_warnings.append(f"Failed to insert document {d_doc_id}: {e}")
                continue

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
                    # get last insert id (best-effort)
                    try:
                        last_html = mariadb_client.run_statement("SELECT LAST_INSERT_ID();")
                        last_val = self._parse_single_result(last_html)
                        last_id = int(last_val) if last_val is not None else None
                    except Exception:
                        last_id = None
                    if last_id is not None:
                        inserted_chunk_ids.append((idx, last_id))
                    else:
                        try:
                            sel_html = mariadb_client.run_statement(
                                f"SELECT id FROM `{dbname}`.`chunks` WHERE doc_id = {self._sql_escape(d_doc_id)} AND chunk_index = {idx} LIMIT 1;"
                            )
                            sel_val = self._parse_single_result(sel_html)
                            inserted_chunk_ids.append((idx, int(sel_val)) if sel_val is not None else (idx, None))
                        except Exception:
                            inserted_chunk_ids.append((idx, None))
                except Exception as e:
                    user_warnings.append(f"Failed to insert chunk {idx} for {d_doc_id}: {e}")
                    inserted_chunk_ids.append((idx, None))
                    continue

            # embeddings: compute and insert (native if allowed, else JSON)
            if chunks:
                embs = self._embed_batch(chunks, embedding_dim)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs_norm = (embs / norms).astype(np.float32)

                for (i, chunk_db_id), vec in zip(inserted_chunk_ids, embs_norm):
                    if chunk_db_id is None:
                        user_warnings.append(f"No chunk id for doc {d_doc_id} chunk {i}; embedding skipped.")
                        continue

                    vec_list = [float(v) for v in vec.tolist()]
                    vec_literal = "[" + ",".join(repr(x) for x in vec_list) + "]"

                    if not use_native_vector:
                        # always use JSON fallback
                        try:
                            emb_json_literal = self._sql_escape(json.dumps(vec_list))
                            mariadb_client.run_statement(
                                f"""
                                INSERT INTO `{dbname}`.`embeddings_json` (chunk_id, model, dim, embedding_json)
                                VALUES ({chunk_db_id}, {self._sql_escape('all-MiniLM-L6-v2')}, {embedding_dim}, {emb_json_literal})
                                ON DUPLICATE KEY UPDATE model=VALUES(model), dim=VALUES(dim), embedding_json=VALUES(embedding_json);
                                """
                            )
                            # verify
                            try:
                                verify_json = mariadb_client.run_statement(
                                    f"SELECT COUNT(*) FROM `{dbname}`.`embeddings_json` WHERE chunk_id = {chunk_db_id};"
                                )
                                cnt = self._parse_single_result(verify_json)
                                if cnt and int(cnt) > 0:
                                    fallback_successes += 1
                                    total_emb_rows += 1
                                else:
                                    fallback_failures += 1
                            except Exception:
                                fallback_failures += 1
                        except Exception as e_json:
                            user_warnings.append(f"Fallback embedding storage failed for chunk_id={chunk_db_id}: {e_json}")
                            fallback_failures += 1
                        continue

                    # Attempt native VECTOR insert
                    native_attempts += 1
                    try:
                        mariadb_client.run_statement(
                            f"""
                            INSERT INTO `{dbname}`.`embeddings` (chunk_id, model, dim, embedding_vector)
                            VALUES ({chunk_db_id}, {self._sql_escape('all-MiniLM-L6-v2')}, {embedding_dim}, {vec_literal})
                            ON DUPLICATE KEY UPDATE model=VALUES(model), dim=VALUES(dim), embedding_vector=VALUES(embedding_vector);
                            """
                        )
                    except Exception as e_native:
                        native_failures += 1
                        # try fallback JSON
                        try:
                            emb_json_literal = self._sql_escape(json.dumps(vec_list))
                            mariadb_client.run_statement(
                                f"""
                                INSERT INTO `{dbname}`.`embeddings_json` (chunk_id, model, dim, embedding_json)
                                VALUES ({chunk_db_id}, {self._sql_escape('all-MiniLM-L6-v2')}, {embedding_dim}, {emb_json_literal})
                                ON DUPLICATE KEY UPDATE model=VALUES(model), dim=VALUES(dim), embedding_json=VALUES(embedding_json);
                                """
                            )
                            try:
                                verify_json = mariadb_client.run_statement(
                                    f"SELECT COUNT(*) FROM `{dbname}`.`embeddings_json` WHERE chunk_id = {chunk_db_id};"
                                )
                                cnt = self._parse_single_result(verify_json)
                                if cnt and int(cnt) > 0:
                                    fallback_successes += 1
                                    total_emb_rows += 1
                                else:
                                    fallback_failures += 1
                            except Exception:
                                fallback_failures += 1
                        except Exception as e_json:
                            user_warnings.append(f"Fallback embedding storage failed for chunk_id={chunk_db_id}: {e_json}")
                            fallback_failures += 1
                        continue

                    # Verify native insert succeeded by COUNT(*)
                    try:
                        verify = mariadb_client.run_statement(
                            f"SELECT COUNT(*) FROM `{dbname}`.`embeddings` WHERE chunk_id = {chunk_db_id};"
                        )
                        cnt = self._parse_single_result(verify)
                        if cnt and int(cnt) > 0:
                            native_successes += 1
                            total_emb_rows += 1
                        else:
                            # fallback to JSON
                            native_failures += 1
                            try:
                                emb_json_literal = self._sql_escape(json.dumps(vec_list))
                                mariadb_client.run_statement(
                                    f"""
                                    INSERT INTO `{dbname}`.`embeddings_json` (chunk_id, model, dim, embedding_json)
                                    VALUES ({chunk_db_id}, {self._sql_escape('all-MiniLM-L6-v2')}, {embedding_dim}, {emb_json_literal})
                                    ON DUPLICATE KEY UPDATE model=VALUES(model), dim=VALUES(dim), embedding_json=VALUES(embedding_json);
                                    """
                                )
                                try:
                                    verify_json = mariadb_client.run_statement(
                                        f"SELECT COUNT(*) FROM `{dbname}`.`embeddings_json` WHERE chunk_id = {chunk_db_id};"
                                    )
                                    cntj = self._parse_single_result(verify_json)
                                    if cntj and int(cntj) > 0:
                                        fallback_successes += 1
                                        total_emb_rows += 1
                                    else:
                                        fallback_failures += 1
                                except Exception:
                                    fallback_failures += 1
                            except Exception as e_json:
                                user_warnings.append(f"Fallback embedding storage failed for chunk_id={chunk_db_id}: {e_json}")
                                fallback_failures += 1
                    except Exception as e_verify:
                        user_warnings.append(f"Verify select for embeddings failed: {e_verify}")

        # Final diagnostics: counts & version
        try:
            cnt_emb = mariadb_client.run_statement("SELECT COUNT(*) FROM embeddings;")
            cnt_emb_val = self._parse_single_result(cnt_emb) or "0"
        except Exception:
            cnt_emb_val = "N/A"
        try:
            cnt_json = mariadb_client.run_statement("SELECT COUNT(*) FROM embeddings_json;")
            cnt_json_val = self._parse_single_result(cnt_json) or "0"
        except Exception:
            cnt_json_val = "N/A"
        try:
            version = mariadb_client.run_statement("SELECT VERSION();")
            version_val = self._parse_single_result(version) or ""
        except Exception:
            version_val = ""

        # concise output
        kernel._send_message("stdout", (
            "Ingest complete.\n"
            f" documents={len(docs_to_ingest)}\n"
            f" chunks_total={total_chunks}\n"
            f" embeddings_written={total_emb_rows}\n"
            f" Server version: {version_val}\n"
        ))

        # if user_warnings:
        #     kernel._send_message("stderr", "Warnings/notes:\n")
        #     for w in user_warnings:
        #         kernel._send_message("stderr", f" - {w}\n")

        return
