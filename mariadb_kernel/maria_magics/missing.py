"""This class implements the %missing magic command

The %missing magic computes NULL / missing-value counts (and percent)
for every column returned by a SQL SELECT or for all columns of a
given table. It mirrors the style of other magics in this package
(e.g. %df) and uses the kernel messaging API for output.

Usage (line magic):
    > %missing              # operates on the DataFrame stored in data["last_select"]
    > %missing my_schema.my_table
    > %missing my_table
    > %missing "SELECT id, email, last_login FROM my_table WHERE dt > '2025-01-01'"

Notes:
- When run without an argument, this magic expects a pandas DataFrame
  available as data["last_select"] (same model as %df).
- When given a table name or SELECT, the magic will try several common
  kernel interfaces to execute SQL. If none exist, it prints instructions.
"""

# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.
from mariadb_kernel.maria_magics.line_magic import LineMagic
import pandas as pd
import re
import traceback


help_text = """
The %missing magic command has the following syntax:
    > %missing [TABLE_OR_SELECT]

If no argument is specified, the magic computes missing value counts
for the result of the last SELECT executed in the notebook (data["last_select"]).
If an argument is given it can be either:
  - a table name: schema.table or table
  - a SELECT statement: SELECT ... (wrap in quotes if your shell strips spaces)

Output: a small table with columns: column, null_count, null_fraction.
"""


class Missing(LineMagic):
    def __init__(self, arg):
        """
        Accept a single constructor argument (keeps parity with other magics).
        `arg` is the string following %missing on the line (may be empty).
        """
        # follow DF style: argument provided by the parser (may be "")
        self.arg = arg.strip() if isinstance(arg, str) else ""
        # nothing else to initialize

    def name(self):
        return "%missing"

    def type(self):
        # lsmagic expects 'line' (matching registry keys)
        return "line"

    def help(self):
        return help_text

    # --- Helpers -------------------------------------------------------
    def _is_select(self, s: str) -> bool:
        return bool(re.match(r"(?is)^\s*select\b", s))

    def _run_sql_via_kernel(self, kernel, sql: str):
        """
        Best-effort: try a few common kernel interfaces to execute SQL and
        return a pandas DataFrame. If execution fails, raise RuntimeError.
        """
        sql = sql.strip().rstrip(";")
        # 1) If kernel exposes a convenience method that returns a DataFrame
        for method_name in ("execute_sql", "run_sql", "execute", "_execute", "_run_sql"):
            try:
                method = getattr(kernel, method_name, None)
                if callable(method):
                    res = method(sql)
                    # If it returned a pandas DataFrame already
                    if isinstance(res, pd.DataFrame):
                        return res
                    # If it returned list of dicts
                    if isinstance(res, list) and res and isinstance(res[0], dict):
                        return pd.DataFrame(res)
                    # If it returned (rows, desc) - try to turn to DataFrame
                    if isinstance(res, tuple) and len(res) >= 2:
                        rows, desc = res[0], res[1]
                        # desc may be DB-API description; try to get column names
                        try:
                            cols = [d[0] for d in desc] if desc else None
                            return pd.DataFrame(rows, columns=cols)
                        except Exception:
                            return pd.DataFrame(rows)
                    # If it's rows only (list of tuples)
                    if isinstance(res, list):
                        return pd.DataFrame(res)
            except Exception:
                # ignore and try next
                pass

        # 2) Try DB-API style cursor on kernel (cursor or connection attributes)
        try:
            if hasattr(kernel, "cursor"):
                cur = kernel.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                desc = cur.description
                cols = [d[0] for d in desc] if desc else None
                return pd.DataFrame(rows, columns=cols)
        except Exception:
            pass

        for conn_attr in ("connection", "conn", "db"):
            try:
                conn = getattr(kernel, conn_attr, None)
                if conn is not None:
                    cur = conn.cursor()
                    cur.execute(sql)
                    rows = cur.fetchall()
                    desc = cur.description
                    cols = [d[0] for d in desc] if desc else None
                    return pd.DataFrame(rows, columns=cols)
            except Exception:
                pass

        # Nothing worked
        raise RuntimeError(
            "Could not execute SQL: kernel does not expose a recognized execution API. "
            "Try running the SELECT first (so data['last_select'] exists) and then call %missing "
            "with no arguments, or adapt this magic to your kernel's execution method."
        )

    def _format_missing_table_str(self, df_missing: pd.DataFrame) -> str:
        """
        Accepts a DataFrame with index=column names and columns ['null_count', 'null_fraction'].
        Returns a formatted string suitable for sending to stdout.
        """
        # Reset index for a cleaner table
        out = df_missing.reset_index().rename(columns={"index": "column"})
        # Format null_fraction as percentage with 2 decimal places
        out["null_fraction"] = (out["null_fraction"] * 100).map(lambda v: f"{v:.2f}%")
        # Use pandas pretty printing
        return out.to_string(index=False)

    # --- Main entry point ---------------------------------------------
    def execute(self, kernel, data):
        """
        Execute the %missing magic.

        kernel: kernel instance (mariadb_kernel.kernel)
        data: dictionary-like object provided by kernel; we expect data["last_select"]
              to be a pandas DataFrame when no argument is provided.
        """
        try:
            arg = self.arg

            # 1) No argument: operate on last_select DataFrame (same as %df)
            if not arg:
                if "last_select" not in data:
                    kernel._send_message("stderr", "No previous SELECT result found (data['last_select'] missing).")
                    return
                df = data["last_select"]
                # data["last_select"] should be a pandas DataFrame
                if not isinstance(df, pd.DataFrame):
                    kernel._send_message("stderr", "data['last_select'] is not a pandas DataFrame.")
                    return
                if df.empty:
                    kernel._send_message("stderr", "There is no query previously executed. No data to analyze.")
                    return

            else:
                # 2) Argument provided - either a SELECT or a table identifier.
                # If it's a SELECT, run it. If it looks like a table name, query all columns.
                if self._is_select(arg):
                    # run the select and get DataFrame
                    df = self._run_sql_via_kernel(kernel, arg)
                else:
                    # treat as table name: possibly schema.table or table
                    table_raw = arg.strip().strip("`").strip()
                    # Build a simple SELECT * FROM table LIMIT 0 to obtain columns and then compute counts
                    # But for counting missing values we need the data; we'll select all columns but compute counts server-side
                    # Simpler: SELECT * FROM table
                    select_sql = f"SELECT * FROM {table_raw}"
                    df = self._run_sql_via_kernel(kernel, select_sql)

                if not isinstance(df, pd.DataFrame):
                    kernel._send_message("stderr", "Query execution did not return a pandas DataFrame.")
                    return

                if df.empty:
                    # it's possible the table is empty — that's fine
                    kernel._send_message("stdout", f"The result for '{arg}' is empty (no rows). All NULL counts are 0.")
                    return

            # At this point `df` is a DataFrame with the data to inspect
            # Compute missing counts and fractions
            null_counts = df.isnull().sum()
            total = len(df)
            null_frac = null_counts / total if total > 0 else 0

            missing_df = pd.DataFrame(
                {"null_count": null_counts.astype(int), "null_fraction": null_frac}
            ).sort_values(by="null_count", ascending=False)

            out_str = self._format_missing_table_str(missing_df)

            # Send result to stdout so it appears in the notebook output
            header = f"Missing-value summary (rows: {total})"
            full_msg = f"{header}\n{out_str}"
            kernel._send_message("stdout", full_msg)

        except Exception as e:
            # Print traceback to assist debugging (consistent with other magics)
            tb = traceback.format_exc()
            kernel._send_message("stderr", f"Error while computing missing values:\n{str(e)}\n{tb}")
            # do not re-raise here; magics usually report errors via stderr
