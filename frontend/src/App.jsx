import React, { useState } from "react";
import "./index.css";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSql, setShowSql] = useState(false);
  const [showPlan, setShowPlan] = useState(false);

  async function ask(e) {
    e && e.preventDefault();
    setLoading(true);
    setResponse(null);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const j = await res.json();
      setResponse(j);
    } catch (err) {
      setResponse({ narrative: "Error: " + err.message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-5xl mx-auto bg-white rounded-2xl shadow p-6">
        <h1 className="text-2xl font-bold mb-4">AI Data Agent — Interview Demo</h1>

        <form onSubmit={ask} className="mb-4">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a complex business question, e.g. 'Show monthly revenue by product category for the last 12 months and explain anomalies.'"
            className="w-full p-3 border rounded mb-2"
            rows={3}
          />
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-blue-600 text-white rounded" disabled={loading}>
              {loading ? "Analyzing..." : "Ask"}
            </button>
            <button type="button" className="px-4 py-2 border rounded" onClick={() => { setQuestion(""); setResponse(null); }}>
              Reset
            </button>
            <label className="ml-auto flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showSql} onChange={(e) => setShowSql(e.target.checked)} /> Show generated SQL
            </label>
          </div>
        </form>

        <div className="space-y-4">
          <div>
            <h2 className="font-semibold">Answer (natural language)</h2>
            <div className="mt-2 p-4 bg-slate-100 rounded">
              <div style={{ whiteSpace: "pre-wrap" }}>{response?.narrative ?? <em>Ask a question to get started.</em>}</div>
            </div>
          </div>

          {showSql && response?.generated_sql && (
            <div>
              <h3 className="font-semibold">Generated SQL</h3>
              <pre className="mt-2 bg-white border rounded p-3 text-xs overflow-auto">{response.generated_sql}</pre>
            </div>
          )}

          {response?.plan && (
            <div>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Plan</h3>
                <button className="text-sm underline" onClick={() => setShowPlan(!showPlan)}>{showPlan ? "Hide" : "Show"} plan</button>
              </div>
              {showPlan && (
                <div className="mt-2 space-y-2">
                  {response.plan.map((p) => (
                    <div key={p.step} className="p-3 bg-white border rounded text-sm">
                      <div className="font-medium">Step {p.step}: {p.action}</div>
                      <div className="text-xs text-slate-600">{p.note}</div>
                      {p.sql && <pre className="mt-2 text-xs overflow-auto bg-slate-50 p-2 rounded">{p.sql}</pre>}
                      {p.rows_preview && p.rows_preview.length > 0 && (
                        <div className="mt-2 text-xs">Preview rows: {JSON.stringify(p.rows_preview.slice(0, 2))}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {response?.table && response.table.length > 0 && (
            <div>
              <h3 className="font-semibold">Table (preview)</h3>
              <div className="mt-2 overflow-auto border rounded">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-100">
                    <tr>
                      {Object.keys(response.table[0]).map((col) => (<th key={col} className="px-3 py-2 text-left">{col}</th>))}
                    </tr>
                  </thead>
                  <tbody>
                    {response.table.slice(0, 50).map((r, idx) => (
                      <tr key={idx} className={idx % 2 ? "bg-white" : "bg-slate-50"}>
                        {Object.keys(r).map((c) => <td key={c} className="px-3 py-2">{String(r[c])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {response?.chart && response.chart.length > 0 && (
            <div>
              <h3 className="font-semibold">Chart</h3>
              <div style={{ height: 340 }} className="mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={response.chart}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="y" stroke="#0ea5e9" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

        </div>

        <footer className="mt-6 text-xs text-slate-500">Demo • React + FastAPI • Generated by the AI Data Agent</footer>
      </div>
    </div>
  );
}
