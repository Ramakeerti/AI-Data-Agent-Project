# backend/agent.py
import os
import re
import json
import time
import duckdb
import pandas as pd
from typing import List, Dict, Any

# Optional: OpenAI client. If you want LLM-driven planning, set OPENAI_API_KEY in env.
try:
    import openai
    OPENAI_AVAILABLE = True
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except Exception:
    OPENAI_AVAILABLE = False

DB_PATH = os.environ.get("DUCKDB_PATH", "backend/data/analytics.duckdb")
ROW_LIMIT = 5000

BANNED_PATTERNS = [r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b", r"\bALTER\b", r"\bVACUUM\b", r"\bATTACH\b", r"\bDETACH\b", r";--", r"--\s"]
BANNED_SYS = [r"information_schema", r"pg_", r"pg_catalog"]

class Agent:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = duckdb.connect(db_path)

    def introspect(self, sample_rows: int = 3) -> Dict[str, Any]:
        tables = {}
        for r in self.conn.execute("SHOW TABLES").fetchall():
            t = r[0]
            try:
                desc = self.conn.execute(f"DESCRIBE {t}").fetchall()
                cols = [c[0] for c in desc]
            except Exception:
                df = self.conn.execute(f"SELECT * FROM {t} LIMIT 1").fetchdf()
                cols = list(df.columns)
            sample_df = self.conn.execute(f"SELECT * FROM {t} LIMIT {sample_rows}").fetchdf()
            sample = sample_df.to_dict(orient="records")
            tables[t] = {"columns": cols, "sample": sample}
        return tables

    def _ban_check(self, sql: str):
        U = sql.upper()
        for pat in BANNED_PATTERNS + BANNED_SYS:
            if re.search(pat, U, flags=re.IGNORECASE):
                raise ValueError(f"Forbidden token or pattern detected in SQL: {pat}")

    def _ensure_limit(self, sql: str) -> str:
        if re.search(r"(?i)LIMIT\s+\d+", sql):
            return sql
        return sql.strip().rstrip(";") + f" LIMIT {ROW_LIMIT}"

    def sanitize_sql(self, sql: str) -> str:
        if not re.match(r"(?is)^(WITH\b|SELECT\b)", sql.strip()):
            raise ValueError("Only read-only SELECT queries or CTEs are allowed.")
        self._ban_check(sql)
        safe_sql = self._ensure_limit(sql)
        return safe_sql

    def call_llm_for_plan(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        If OpenAI key available, call LLM to produce a structured plan (JSON).
        Otherwise return a heuristic deterministic plan for demo.
        """
        grounding = []
        for t, m in schema.items():
            grounding.append(f"Table {t}: columns = {', '.join(m['columns'])}; samples = {json.dumps(m['sample'][:2], default=str)}")
        grounding_text = "\n".join(grounding)[:3500]

        if OPENAI_AVAILABLE:
            system = (
                "You are a SQL planning assistant. Given schema grounding and a user question, "
                "return valid JSON only: { \"plan\": [ {\"step\":1, \"action\":\"clean\", \"sql\":\"...\", \"note\":\"...\"}, ... ] }.\n"
                "Each SQL must be a read-only SELECT compatible with DuckDB. If unsure about column names, pick a plausible fallback and explain in note."
            )
            user = f"Schema:\n{grounding_text}\n\nQuestion: {question}\n\nReturn JSON with plan."
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=800
                )
                text = resp["choices"][0]["message"]["content"].strip()
                m = re.search(r"\{(?:.|\n)*\}", text)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    parsed = json.loads(text)
                return parsed
            except Exception as e:
                # fallback to heuristic
                print("LLM call failed, falling back to heuristic:", e)

        # Heuristic fallback plan (common analytics: monthly revenue by category)
        # Choose first table that has amount-like column
        chosen_table = None
        chosen_amount = None
        chosen_date = None
        for t, m in schema.items():
            for c in m["columns"]:
                if re.search(r"amount|amt|price|total|revenue", c, re.IGNORECASE):
                    chosen_table = t
                    chosen_amount = c
            for c in m["columns"]:
                if re.search(r"date|created|ts|time|month", c, re.IGNORECASE):
                    chosen_date = c
        if not chosen_table:
            chosen_table = next(iter(schema.keys()))
            chosen_amount = schema[chosen_table]["columns"][0]
        if not chosen_date:
            cols = schema[chosen_table]["columns"]
            chosen_date = cols[1] if len(cols) > 1 else cols[0]

        plan = []
        # step 1: preview and cleaning
        plan.append({
            "step": 1,
            "action": "preview_and_clean",
            "sql": f"SELECT {chosen_date} AS date_raw, {chosen_amount} AS amount_raw, * FROM {chosen_table} LIMIT 200",
            "note": "Preview raw values for date and amount columns."
        })
        # step 2: aggregate after cleaning (replace commas in amount, cast)
        plan.append({
            "step": 2,
            "action": "aggregate_monthly",
            "sql": (
                "WITH cleaned AS (\n"
                f"  SELECT TRY_CAST(REPLACE({chosen_amount}, ',', '') AS DOUBLE) AS amount, TRY_CAST({chosen_date} AS DATE) AS dt, *\n"
                f"  FROM {chosen_table}\n"
                ")\n"
                "SELECT STRFTIME('%Y-%m', dt) AS month,  COALESCE(prod_cat, 'Unknown') AS category, SUM(amount) AS total_amount\n"
                "FROM cleaned\n"
                "GROUP BY month, category\n"
                "ORDER BY month"
            ),
            "note": "Sum amounts by month and category. Replace commas and cast to numeric."
        })
        # step 3: post-analysis (anomaly detection in Python)
        plan.append({
            "step": 3,
            "action": "analyze_anomalies",
            "sql": None,
            "note": "Detect anomalies with z-score and generate chart payload."
        })

        return {"plan": plan}

    def make_chart(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        # Expect rows ordered by month and may include category breakdown
        # Build a time series by grouping month into x and sum of total_amount into y (if provided)
        # If rows have 'month' and 'total_amount', use them.
        if "month" in rows[0] and ("total_amount" in rows[0] or "amount" in rows[0]):
            chart = []
            for r in rows:
                y = r.get("total_amount") or r.get("amount") or 0
                chart.append({"x": r.get("month"), "y": float(y) if y is not None else 0})
            return chart
        # fallback: index-y
        chart = []
        for i, r in enumerate(rows):
            # find numeric
            val = None
            for v in r.values():
                if isinstance(v, (int, float)):
                    val = v
                    break
                try:
                    v2 = float(str(v).replace(",", ""))
                    val = v2
                    break
                except:
                    pass
            chart.append({"x": i, "y": float(val) if val is not None else 0})
        return chart

    def detect_anomalies(self, series: List[float], z_thresh: float = 2.5) -> List[int]:
        if not series:
            return []
        mean = float(sum(series) / len(series))
        import math
        variance = sum((x - mean) ** 2 for x in series) / len(series)
        sd = math.sqrt(variance)
        out = []
        for i, v in enumerate(series):
            if sd == 0:
                z = 0
            else:
                z = (v - mean) / sd
            if abs(z) >= z_thresh:
                out.append(i)
        return out

    def answer(self, question: str) -> Dict[str, Any]:
        start = time.time()
        schema = self.introspect()
        plan_obj = self.call_llm_for_plan(question, schema)
        plan = plan_obj.get("plan", [])

        plan_steps = []
        final_table = []
        chart_payload = []
        generated_sqls = []

        # Execute planned SQL steps
        for step in plan:
            step_no = step.get("step")
            action = step.get("action")
            sql = step.get("sql")
            note = step.get("note", "")
            rows_preview = []
            try:
                if sql:
                    safe = self.sanitize_sql(sql)
                    df = self.conn.execute(safe).fetchdf()
                    rows_preview = df.to_dict(orient="records")
                    generated_sqls.append(sql)
                plan_steps.append({"step": step_no, "action": action, "note": note, "sql": sql, "rows_preview": rows_preview})
            except Exception as e:
                plan_steps.append({"step": step_no, "action": action, "note": f"Failed: {e}", "sql": sql, "rows_preview": []})

        # If there is an aggregation step return those results and run anomalies
        # find last step with aggregate_monthly or similar
        agg_df = None
        for p in plan_steps:
            if p["action"] in ("aggregate_monthly", "aggregate", "group_by_month"):
                if p["rows_preview"]:
                    agg_df = pd.DataFrame(p["rows_preview"])
                    break
        if agg_df is None:
            # fall back to last preview with rows
            for p in reversed(plan_steps):
                if p["rows_preview"]:
                    agg_df = pd.DataFrame(p["rows_preview"])
                    break

        narrative = ""
        try:
            if agg_df is not None and not agg_df.empty:
                # If we have month + category + total_amount, prepare time series summary
                if "month" in agg_df.columns and "total_amount" in agg_df.columns:
                    # pivot by month summing total_amount
                    pivot = agg_df.groupby("month")["total_amount"].sum().reset_index().sort_values("month")
                    series = pivot["total_amount"].astype(float).tolist()
                    anomaly_idxs = self.detect_anomalies(series)
                    chart_payload = [{"x": r["month"], "y": float(r["total_amount"]), "anomaly": (i in anomaly_idxs)} for i, r in pivot.to_dict(orient="records")]
                    final_table = agg_df.to_dict(orient="records")
                    narrative = f"I aggregated revenue by month. Found {len(anomaly_idxs)} anomalous month(s) using z-score. Top months by revenue: {pivot.sort_values('total_amount', ascending=False).head(3).to_dict(orient='records')}."
                else:
                    # fallback: show head
                    final_table = agg_df.to_dict(orient="records")[:200]
                    chart_payload = self.make_chart(final_table)
                    narrative = f"Returned aggregated preview with {len(final_table)} rows."
            else:
                narrative = "I couldn't produce aggregated results from the plan. Showing previews if available."
                final_table = plan_steps[-1]["rows_preview"] if plan_steps else []
                chart_payload = self.make_chart(final_table)
        except Exception as e:
            narrative = f"Analysis failed: {e}"
            final_table = []
            chart_payload = []

        total_time = time.time() - start
        narrative = f"{narrative} (Plan executed in {total_time:.2f}s)."

        return {
            "narrative": narrative,
            "table": final_table,
            "chart": chart_payload,
            "plan": plan_steps,
            "generated_sql": (generated_sqls[-1] if generated_sqls else "")
        }
