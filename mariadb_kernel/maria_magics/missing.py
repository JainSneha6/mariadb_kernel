# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import re
import traceback

try:
    # optional niceties
    from tabulate import tabulate  # nice table output if available
except Exception:
    tabulate = None


class Missing(MariaMagic):
    """
    Line magic for computing missing (NULL) counts per column.

    Usage (line magic):
      %missing my_schema.my_table
      %missing my_table
      %missing SELECT id, name, val FROM ... WHERE ...
    """

    def name(self):
        return "missing"

    def type(self):
        return "line"

    def help(self):
        return (
            "Usage: %missing <table> | %missing <SELECT ...>\n\n"
            "If you provide a table name (optionally schema-qualified) the magic\n"
            "will query INFORMATION_SCHEMA for the columns and compute NULL counts\n"
            "for each column. If you provide a SELECT statement it will compute\n"
            "NULL counts on the result of that SELECT (the SELECT is executed as\n"
            "a subquery)."
        )

    # --- helpers ---------------------------------------------------------
    def _clean_sql(self, sql):
        # remove trailing semicolons and whitespace
        return sql.strip().rstrip(";")

    def _run_sql(self, kernel, sql):
        """
        Try several common ways to run SQL against the kernel and return
        (rows, description) where description is DB-API cursor.description-like
        (sequence of (name, ...)).
        """
        sql = self._clean_sql(sql)
        # Try kernel.cursor() pattern (DB-API)
        try:
            if hasattr(kernel, "cursor"):
                cur = kernel.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                desc = cur.description
                return rows, desc
        except Exception:
            # fall through and try other approaches
            pass

        # Try kernel.connection / kernel.conn
        for conn_attr in ("connection", "conn", "db"):
            try:
                conn = getattr(kernel, conn_attr, None)
                if conn is not None:
                    cur = conn.cursor()
                    cur.execute(sql)
                    rows = cur.fetchall()
                    desc = cur.description
                    return rows, desc
            except Exception:
                pass

        # Try a convenience method the kernel might expose
        for method_name in ("execute_sql", "run_sql", "_run_sql", "_execute", "execute"):
            try:
                method = getattr(kernel, method_name, None)
                if callable(method):
                    # many kernel helpers return results in different formats;
                    # attempt to normalize
                    res = method(sql)
                    # If it's a tuple (rows, desc) return directly
                    if isinstance(res, tuple) and len(res) >= 2:
                        return res[0], res[1]
                    # If it's list-of-dicts -> derive description
                    if isinstance(res, list) and res and isinstance(res[0], dict):
                        rows = [tuple(r.values()) for r in res]
                        desc = tuple((k,) + (None,) * 6 for k in res[0].keys())
                        return rows, desc
                    # If it's a pandas DataFrame
                    try:
                        import pandas as pd

                        if isinstance(res, pd.DataFrame):
                            rows = [tuple(x) for x in res.to_records(index=False)]
                            desc = tuple((c,) + (None,) * 6 for c in res.columns)
                            return rows, desc
                    except Exception:
                        pass
                    # otherwise, try to interpret as rows-only
                    if isinstance(res, list):
                        desc = None
                        return res, desc
            except Exception:
                pass

        # Last resort: raise
        raise RuntimeError("Couldn't execute SQL: no suitable kernel DB API found.")

    def _get_column_names_from_select(self, kernel, select_sql):
        """
        Execute SELECT ... LIMIT 0 to obtain column names via cursor.description
        """
        # Wrap the user's select as a derived table and limit 0
        wrapper = f"SELECT * FROM ({self._clean_sql(select_sql)}) AS _sub LIMIT 0"
        try:
            _, desc = self._run_sql(kernel, wrapper)
            if desc:
                return [d[0] for d in desc]
            # If description not available, try executing the original and infer keys
            rows, desc2 = self._run_sql(kernel, self._clean_sql(select_sql) + " LIMIT 1")
            if desc2:
                return [d[0] for d in desc2]
        except Exception:
            # propagate a helpful error upward
            raise
        return []

    def _get_columns_for_table(self, kernel, schema, table):
        """
        Query INFORMATION_SCHEMA to list columns for a given schema.table.
        """
        schema = schema or self._get_current_database(kernel)
        sql = (
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}' "
            "ORDER BY ORDINAL_POSITION"
        )
        rows, desc = self._run_sql(kernel, sql)
        cols = [r[0] for r in rows] if rows else []
        return cols

    def _get_current_database(self, kernel):
        """Try SELECT DATABASE()"""
        try:
            rows, desc = self._run_sql(kernel, "SELECT DATABASE()")
            if rows and len(rows) >= 1:
                # row can be ('db',) or [ ('db',) ]
                first = rows[0]
                if isinstance(first, tuple) or isinstance(first, list):
                    return first[0]
                return first
        except Exception:
            pass
        return None

    def _format_and_print(self, result_pairs):
        """
        result_pairs: list of (column_name, null_count)
        Print in a readable table; return string too.
        """
        if tabulate:
            table_str = tabulate(result_pairs, headers=("column", "null_count"), tablefmt="github")
            print(table_str)
            return table_str

        # fallback
        max_col = max((len(str(r[0])) for r in result_pairs), default=6)
        header = f"{'column'.ljust(max_col)} | null_count"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for col, cnt in result_pairs:
            print(f"{str(col).ljust(max_col)} | {cnt}")
        return header

    # --- main entry point -----------------------------------------------
    def execute(self, kernel, data):
        """
        Execute the magic.
        kernel: the kernel instance (object provided by mariadb_kernel)
        data: str - what's after the %missing on the line
        """
        raw = data.strip()
        if not raw:
            print(self.help())
            return

        try:
            # Distinguish a SELECT vs a table name (very simple heuristic)
            is_select = bool(re.match(r"(?is)^\s*select\b", raw))

            if is_select:
                user_select = raw
                # get column names from the select
                cols = self._get_column_names_from_select(kernel, user_select)
                if not cols:
                    print("No columns found for the provided SELECT.")
                    return

                # build aggregate query that counts NULLs per column
                aggregates = ", ".join(
                    [f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in cols]
                )
                agg_sql = f"SELECT {aggregates} FROM ({self._clean_sql(user_select)}) AS _sub"
                rows, desc = self._run_sql(kernel, agg_sql)

                if rows and len(rows) >= 1:
                    row = rows[0]
                    pairs = list(zip(cols, row))
                    # print readable output
                    self._format_and_print(pairs)
                else:
                    print("Query returned no rows; NULL counts are zero or query failed.")
                return

            # Otherwise treat as schema-qualified table or plain table
            # split schema.table
            if "." in raw:
                parts = raw.split(".", 1)
                schema = parts[0].strip(" `")
                table = parts[1].strip(" `")
            else:
                schema = None
                table = raw.strip(" `")

            cols = self._get_columns_for_table(kernel, schema, table)
            if not cols:
                print(f"No columns found for table {raw}.")
                return

            aggregates = ", ".join(
                [f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in cols]
            )
            schema_prefix = f"`{schema}`." if schema else ""
            agg_sql = f"SELECT {aggregates} FROM {schema_prefix}`{table}`"
            rows, desc = self._run_sql(kernel, agg_sql)

            if rows and len(rows) >= 1:
                row = rows[0]
                pairs = list(zip(cols, row))
                self._format_and_print(pairs)
            else:
                print("Query returned no rows; NULL counts are zero or query failed.")
        except Exception as e:
            print("Error while computing missing values:")
            traceback.print_exc()
            print(str(e))
