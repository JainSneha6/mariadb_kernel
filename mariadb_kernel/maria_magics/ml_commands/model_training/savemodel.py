import os
import tempfile
import datetime
import pickle
import joblib
import shlex
import json
from distutils import util

from mariadb_kernel.maria_magics.maria_magic import MariaMagic

class SaveModel(MariaMagic):
    """
    %savemodel [model_name=last_model] [save_path=/path/to/model.joblib]
               [db_table=<table>] [db_conn_key=mariadb_conn] [db_uri=<sqlalchemy-uri>]
               [db_host=...] [db_user=...] [db_password=...] [db_name=...]
               [overwrite=True|False] [auto_db=True]

    Save a trained model (stored in data[model_name]) to disk or to a MariaDB table as a BLOB.

    This magic will attempt to automatically detect the active DB connection from:
      - common keys in `data`: mariadb_conn, db_conn, conn, connection, engine, sqlalchemy_engine
      - attributes on the `kernel` object with the same names
      - a connection info dict in data/kernel (e.g. connection_info)
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "savemodel"

    def help(self):
        return "Save trained model to disk or MariaDB storage (auto-detects active DB connection if possible)."

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

    def parse_args(self, input_str):
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _detect_connection(self, kernel, data, preferred_key="mariadb_conn"):
        """
        Try multiple strategies to obtain a DB-API connection or SQLAlchemy engine.
        Returns tuple (conn_obj, cursor_factory, created_conn_bool, info_dict)
          - conn_obj: DB-API connection or SQLAlchemy engine/raw_connection
          - cursor_factory: callable to obtain a cursor from conn_obj (conn_obj.cursor)
          - created_conn_bool: whether this method created the connection (so caller can close it)
          - info_dict: dict with connection metadata (e.g., database name) if found
        """
        # 1) check data for common keys
        keys = [preferred_key, "db_conn", "mariadb_connection", "conn", "connection", "engine", "sqlalchemy_engine", "connection_info"]
        for k in keys:
            if k in data and data[k] is not None:
                obj = data[k]
                # SQLAlchemy engine
                try:
                    from sqlalchemy.engine.base import Engine as _Engine  # type: ignore
                except Exception:
                    _Engine = None
                if _Engine is not None and isinstance(obj, _Engine):
                    try:
                        raw_conn = obj.raw_connection()
                        return raw_conn, (lambda c: c.cursor()), True, {"source": f"data['{k}'] (sqlalchemy engine)"}
                    except Exception:
                        pass
                # DB-API connection-like
                if hasattr(obj, "cursor") and hasattr(obj, "commit"):
                    return obj, (lambda c: c.cursor()), False, {"source": f"data['{k}'] (db-api conn)"}
                # SQLAlchemy connection object (Connection)
                if hasattr(obj, "connection"):
                    try:
                        raw_conn = obj.connection
                        return raw_conn, (lambda c: c.cursor()), True, {"source": f"data['{k}'] (sqlalchemy raw connection)"}
                    except Exception:
                        pass
                # a plain dict of connection params
                if isinstance(obj, dict):
                    return None, None, False, {"conn_params": obj, "source": f"data['{k}'] (params dict)"}

        # 2) check kernel attributes for same keys
        for k in keys + ["mariadb_conn", "db_conn", "connection", "conn", "engine", "sqlalchemy_engine", "current_database", "current_db", "_last_use_db", "connection_info"]:
            if hasattr(kernel, k):
                obj = getattr(kernel, k)
                if obj is None:
                    continue
                try:
                    from sqlalchemy.engine.base import Engine as _Engine  # type: ignore
                except Exception:
                    _Engine = None
                if _Engine is not None and isinstance(obj, _Engine):
                    try:
                        raw_conn = obj.raw_connection()
                        return raw_conn, (lambda c: c.cursor()), True, {"source": f"kernel.{k} (sqlalchemy engine)"}
                    except Exception:
                        pass
                if hasattr(obj, "cursor") and hasattr(obj, "commit"):
                    return obj, (lambda c: c.cursor()), False, {"source": f"kernel.{k} (db-api conn)"}
                if isinstance(obj, dict):
                    return None, None, False, {"conn_params": obj, "source": f"kernel.{k} (params dict)"}

        # 3) try to read a small connection-info dict from common locations
        for info_key in ("connection_info", "conn_info", "db_info"):
            if info_key in data and isinstance(data[info_key], dict):
                return None, None, False, {"conn_params": data[info_key], "source": f"data['{info_key}']"}
            if hasattr(kernel, info_key):
                obj = getattr(kernel, info_key)
                if isinstance(obj, dict):
                    return None, None, False, {"conn_params": obj, "source": f"kernel.{info_key}"}

        # 4) nothing found
        return None, None, False, {}

    def execute(self, kernel, data):
        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        model_name = args.get("model_name", args.get("model", "last_model"))
        save_path = args.get("save_path", None)
        db_table = args.get("db_table", None)
        db_conn_key = args.get("db_conn_key", "mariadb_conn")
        db_uri = args.get("db_uri", None)
        overwrite = bool(args.get("overwrite", False))
        auto_db = bool(args.get("auto_db", True))

        # optional explicit connection details (fallback)
        db_host = args.get("db_host")
        db_user = args.get("db_user")
        db_password = args.get("db_password")
        db_name = args.get("db_name")

        model = data.get(model_name)
        if model is None:
            kernel._send_message("stderr", f"No model found in data['{model_name}']. Train and save a model first.")
            return

        did_something = False

        # Save to disk if requested
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                joblib.dump(model, save_path)
                kernel._send_message("stdout", f"Model saved to {save_path}")
                did_something = True
            except Exception as e:
                kernel._send_message("stderr", f"Failed to save model to disk ({save_path}): {e}")

        # If user asked to save to DB, attempt detection and insert
        if db_table:
            # serialize model to bytes
            try:
                model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                kernel._send_message("stderr", f"Failed to serialize model with pickle: {e}")
                model_bytes = None

            if model_bytes is None:
                kernel._send_message("stderr", "Model serialization failed; cannot save to DB.")
            else:
                conn_obj, cursor_factory, created_conn, info = (None, None, False, {})
                # If db_uri explicitly provided, prefer it (SQLAlchemy)
                if db_uri:
                    try:
                        from sqlalchemy import create_engine, text
                        engine = create_engine(db_uri)
                        raw_conn = engine.raw_connection()
                        conn_obj = raw_conn
                        cursor_factory = (lambda c: c.cursor())
                        created_conn = True
                        info = {"source": "db_uri"}
                    except Exception as e:
                        kernel._send_message("stderr", f"Could not connect via db_uri: {e}")
                        conn_obj = None

                # If auto_db requested, attempt to detect connection from kernel/data
                if conn_obj is None and auto_db:
                    detected_conn, cursor_factory, created_conn_flag, info = self._detect_connection(kernel, data, preferred_key=db_conn_key)
                    conn_obj = detected_conn
                    created_conn = created_conn_flag

                # If detection returned connection params dict, try to open via mariadb connector
                conn_params = info.get("conn_params") if isinstance(info, dict) else None
                if conn_obj is None and conn_params:
                    try:
                        import mariadb
                        # rename keys if necessary
                        host = conn_params.get("host") or conn_params.get("db_host") or conn_params.get("hostaddr")
                        user = conn_params.get("user") or conn_params.get("username")
                        password = conn_params.get("password") or conn_params.get("passwd") or conn_params.get("db_password")
                        database = conn_params.get("database") or conn_params.get("db_name") or conn_params.get("schema")
                        conn_obj = mariadb.connect(host=host, user=user, password=password or "", database=database)
                        cursor_factory = (lambda c: c.cursor())
                        created_conn = True
                        info["source_detail"] = "opened via mariadb from conn_params"
                    except Exception as e:
                        kernel._send_message("stderr", f"Failed to open mariadb connection from params: {e}")
                        conn_obj = None

                # If nothing found yet but explicit host/user provided on command line, try them
                if conn_obj is None and db_host and db_user and db_name:
                    try:
                        import mariadb
                        conn_obj = mariadb.connect(host=db_host, user=db_user, password=db_password or "", database=db_name)
                        cursor_factory = (lambda c: c.cursor())
                        created_conn = True
                        info = {"source": "db_host/db_user arguments"}
                    except Exception as e:
                        kernel._send_message("stderr", f"Could not connect using provided db_host/db_user/db_name: {e}")
                        conn_obj = None

                # Final check: if conn_obj is still None, return helpful error
                if conn_obj is None:
                    kernel._send_message("stderr", "No usable DB connection detected. Provide one via:\n"
                                                 "  - data['mariadb_conn'] (DB-API connection), or\n"
                                                 "  - data['engine'] (SQLAlchemy engine), or\n"
                                                 "  - db_uri=..., or\n"
                                                 "  - db_host/db_user/db_name arguments.\n"
                                                 "Set auto_db=False to suppress detection and provide explicit params.")
                else:
                    # We have a connection-like object (conn_obj) and a cursor factory.
                    inserted = False
                    created_local_conn = created_conn
                    try:
                        # Try to obtain a cursor
                        try:
                            cursor = cursor_factory(conn_obj)
                        except Exception:
                            # fallback: try conn_obj.cursor()
                            try:
                                cursor = conn_obj.cursor()
                            except Exception as e:
                                raise RuntimeError(f"Could not obtain cursor from connection: {e}")

                        # Ensure table exists (simple create)
                        try:
                            create_sql = f"""
                            CREATE TABLE IF NOT EXISTS `{db_table}` (
                                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                                model_name VARCHAR(255),
                                created_at DATETIME,
                                model_blob LONGBLOB
                            )
                            """
                            try:
                                cursor.execute(create_sql)
                            except Exception:
                                # some drivers need different execution path (SQLAlchemy)
                                try:
                                    conn_obj.execute(create_sql)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # If overwrite requested, delete previous with same model_name
                        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            if overwrite:
                                try:
                                    cursor.execute(f"DELETE FROM `{db_table}` WHERE model_name=%s", (model_name,))
                                except Exception:
                                    try:
                                        cursor.execute(f"DELETE FROM `{db_table}` WHERE model_name=:%s", (model_name,))
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # Insert; adapt paramstyle if necessary
                        insert_sql = f"INSERT INTO `{db_table}` (model_name, created_at, model_blob) VALUES (%s, %s, %s)"
                        try:
                            cursor.execute(insert_sql, (model_name, now, model_bytes))
                        except Exception:
                            # try SQLAlchemy style named params
                            try:
                                cursor.execute(insert_sql.replace("%s", ":blob"), {"blob": model_bytes, "model_name": model_name, "created_at": now})
                            except Exception as e:
                                # last resort: use execute with binary literal (unsafe for special bytes) -- avoid
                                raise

                        # commit if method available
                        try:
                            conn_obj.commit()
                        except Exception:
                            pass

                        kernel._send_message("stdout", f"Model stored into DB table '{db_table}' (model_name='{model_name}'). source={info.get('source') or info.get('source_detail', 'detected')}")
                        inserted = True
                        did_something = True
                    except Exception as e:
                        kernel._send_message("stderr", f"Failed to insert model into table '{db_table}': {e}")
                    finally:
                        # close created connections only
                        try:
                            if created_local_conn and conn_obj:
                                try:
                                    cursor.close()
                                except Exception:
                                    pass
                                try:
                                    conn_obj.close()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    if not inserted:
                        kernel._send_message("stderr", f"Model was not inserted into DB table '{db_table}'.")

        if not did_something:
            kernel._send_message("stderr", "No action taken. Provide save_path and/or db_table to save the model.")
        return
