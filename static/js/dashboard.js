/**
 * dashboard.js — Preventum Labs Fraud Intelligence Platform
 * Client-side logic: charts, API calls, transaction history
 */

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  stats: null,
  metrics: null,
  history: [],          // last 20 analyzed transactions
  charts: {},
};

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = {
  num:  (n) => Number(n).toLocaleString(),
  pct:  (n) => `${Number(n).toFixed(2)}%`,
  usd:  (n) => `$${Number(n).toFixed(2)}`,
  score:(n) => `${Number(n).toFixed(1)}%`,
};

function el(id) { return document.getElementById(id); }

function badge(text, cls) {
  return `<span class="badge-${cls} px-2 py-0.5 rounded-full text-xs font-semibold">${text}</span>`;
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function apiFetch(url, opts={}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadStats(), loadMetrics()]);
  startPolling();
  setupForm();
});

// ── Load Stats ────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const stats = await apiFetch("/api/stats");
    state.stats = stats;
    renderKPIs(stats);
    renderDonutChart(stats);
    renderCategoryChart(stats);
    renderHourChart(stats);
  } catch (e) {
    console.error("Stats error:", e);
  }
}

// ── Load Model Metrics ────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const metrics = await apiFetch("/api/model-metrics");
    state.metrics = metrics;
    renderMetrics(metrics);
  } catch (e) {
    console.error("Metrics error:", e);
  }
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function renderKPIs(s) {
  el("kpi-total").textContent    = fmt.num(s.total_transactions);
  el("kpi-fraud").textContent    = fmt.num(s.fraud_count);
  el("kpi-fraud-pct").textContent= `${s.fraud_rate}%`;
  el("kpi-avg-amt").textContent  = fmt.usd(s.avg_amount);

  const acc = state.metrics?.xgb?.accuracy
    ? (state.metrics.xgb.accuracy * 100).toFixed(2) + "%"
    : "—";
  el("kpi-accuracy").textContent = acc;
}

// ── Donut Chart (fraud vs legit) ──────────────────────────────────────────────
function renderDonutChart(s) {
  const ctx = el("donut-chart")?.getContext("2d");
  if (!ctx) return;

  if (state.charts.donut) state.charts.donut.destroy();

  state.charts.donut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fraud", "Legitimate"],
      datasets: [{
        data: [s.fraud_count, s.legitimate_count],
        backgroundColor: ["#ef4444", "#10b981"],
        borderColor: ["#991b1b", "#065f46"],
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      cutout: "70%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", padding: 16, font: { size: 12 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${fmt.num(ctx.raw)} (${((ctx.raw / s.total_transactions)*100).toFixed(2)}%)`,
          },
        },
      },
      animation: { animateRotate: true, duration: 800 },
    },
  });
}

// ── Bar Chart (fraud by category) ─────────────────────────────────────────────
function renderCategoryChart(s) {
  const ctx = el("category-chart")?.getContext("2d");
  if (!ctx) return;

  if (state.charts.category) state.charts.category.destroy();

  const entries = Object.entries(s.fraud_by_category)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  const labels = entries.map(([k]) => k.replace(/_/g, " "));
  const data   = entries.map(([, v]) => v);

  state.charts.category = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Fraud Count",
        data,
        backgroundColor: "rgba(59,130,246,0.7)",
        borderColor: "#3b82f6",
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` ${fmt.num(c.raw)} fraud cases` } },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 10 } },
          grid:  { color: "rgba(255,255,255,0.03)" },
        },
        y: {
          ticks: { color: "#64748b", font: { size: 10 } },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
      },
      animation: { duration: 700 },
    },
  });
}

// ── Line Chart (fraud by hour) ─────────────────────────────────────────────────
function renderHourChart(s) {
  const ctx = el("hour-chart")?.getContext("2d");
  if (!ctx) return;

  if (state.charts.hour) state.charts.hour.destroy();

  const hours  = Array.from({length: 24}, (_, i) => i);
  const counts = hours.map(h => s.fraud_by_hour[String(h)] || 0);

  state.charts.hour = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours.map(h => `${h}:00`),
      datasets: [{
        label: "Fraud Events",
        data: counts,
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245,158,11,0.08)",
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: "#f59e0b",
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => ` ${fmt.num(c.raw)} fraud events` } },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 10 }, maxRotation: 0 },
          grid:  { color: "rgba(255,255,255,0.03)" },
        },
        y: {
          ticks: { color: "#64748b", font: { size: 10 } },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
      },
      animation: { duration: 700 },
    },
  });
}

// ── Model Metrics ─────────────────────────────────────────────────────────────
function renderMetrics(m) {
  const r = (v) => v ? (v * 100).toFixed(1) : "—";
  const models = ["rf", "xgb"];
  const labels = { rf: "Random Forest", xgb: "XGBoost" };

  const container = el("metrics-container");
  if (!container) return;

  container.innerHTML = models.map(key => {
    const met = m[key] || {};
    return `
      <div class="metric-item">
        <div class="text-xs font-semibold text-blue-400 mb-3">${labels[key]}</div>
        <div class="grid grid-cols-2 gap-2">
          ${["accuracy","precision","recall","f1","auc"].map(metric => `
            <div>
              <div class="text-xs text-slate-500 capitalize">${metric}</div>
              <div class="text-sm font-bold text-white">${r(met[metric])}%</div>
            </div>
          `).join("")}
        </div>
      </div>`;
  }).join("");
}

// ── Form Submission ───────────────────────────────────────────────────────────
function setupForm() {
  const form = el("predict-form");
  if (!form) return;
  form.addEventListener("submit", handlePredict);
}

async function handlePredict(e) {
  e.preventDefault();
  const btn = el("analyze-btn");
  const spinner = el("btn-spinner");
  const btnText = el("btn-text");

  btn.disabled = true;
  spinner.classList.remove("hidden");
  btnText.textContent = "Analyzing…";

  try {
    const payload = {
      amt:        parseFloat(el("f-amt").value),
      category:   el("f-category").value,
      city_pop:   parseFloat(el("f-city-pop").value),
      lat:        parseFloat(el("f-lat").value),
      long:       parseFloat(el("f-long").value),
      merch_lat:  parseFloat(el("f-lat").value) + (Math.random() * 0.2 - 0.1),
      merch_long: parseFloat(el("f-long").value) + (Math.random() * 0.2 - 0.1),
      hour:       parseInt(el("f-hour").value),
      gender:     el("f-gender").value,
      age:        parseFloat(el("f-age").value || "35"),
      day_of_week:new Date().getDay(),
      month:      new Date().getMonth() + 1,
    };

    const result = await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderResult(result, payload);
    addToHistory(result, payload);
  } catch (err) {
    el("result-panel").innerHTML = `
      <div class="result-panel result-fraud">
        <div class="text-red-400 font-semibold">⚠ Error: ${err.message}</div>
        <div class="text-sm text-slate-400 mt-1">Make sure the Flask server is running and models are trained.</div>
      </div>`;
    el("result-panel").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "Analyze Transaction";
  }
}

function renderResult(result, payload) {
  const panel = el("result-panel");
  const isFraud = result.is_fraud === 1;
  const cls = isFraud ? "result-fraud" : "result-legit";
  const scoreColor = isFraud
    ? result.confidence > 75 ? "#ef4444" : "#f59e0b"
    : "#10b981";

  panel.innerHTML = `
    <div class="result-panel ${cls}">
      <div class="flex items-center justify-between mb-4">
        <div>
          <div class="text-xs text-slate-400 mb-1">Detection Result</div>
          <div class="text-xl font-bold" style="color:${scoreColor}">
            ${isFraud ? "🚨 FRAUD DETECTED" : "✅ LEGITIMATE"}
          </div>
        </div>
        <div class="text-right">
          <div class="text-3xl font-black" style="color:${scoreColor}">${result.confidence}%</div>
          <div class="text-xs text-slate-400">confidence</div>
        </div>
      </div>

      <div class="confidence-bar mb-4">
        <div class="confidence-fill" style="width:${result.confidence}%; background:${scoreColor}"></div>
      </div>

      <div class="grid grid-cols-3 gap-3 mb-4 text-center">
        <div class="metric-item">
          <div class="text-xs text-slate-500">Risk Level</div>
          <div class="text-sm font-bold mt-1">${badge(result.risk_level, result.risk_level.toLowerCase())}</div>
        </div>
        <div class="metric-item">
          <div class="text-xs text-slate-500">RF Score</div>
          <div class="text-sm font-bold text-white mt-1">${result.rf_score ?? '—'}%</div>
        </div>
        <div class="metric-item">
          <div class="text-xs text-slate-500">XGB Score</div>
          <div class="text-sm font-bold text-white mt-1">${result.xgb_score ?? '—'}%</div>
        </div>
      </div>

      <div>
        <div class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">Risk Factors</div>
        <ul class="space-y-1">
          ${(result.risk_factors || []).map(f =>
            `<li class="text-sm flex items-start gap-2">
              <span style="color:${scoreColor}" class="mt-0.5">•</span>
              <span class="text-slate-300">${f}</span>
            </li>`
          ).join("")}
        </ul>
      </div>
    </div>`;

  panel.classList.remove("hidden");
}

// ── Transaction History ───────────────────────────────────────────────────────
function addToHistory(result, payload) {
  const entry = {
    time:      new Date().toLocaleTimeString(),
    amt:       payload.amt,
    category:  payload.category,
    city_pop:  payload.city_pop,
    score:     result.confidence,
    risk:      result.risk_level,
    is_fraud:  result.is_fraud,
  };
  state.history.unshift(entry);
  if (state.history.length > 20) state.history.pop();
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = el("history-tbody");
  if (!tbody) return;

  if (state.history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center text-slate-500 py-8 text-sm">No transactions analyzed yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = state.history.map(tx => {
    const rowCls = tx.is_fraud ? "row-fraud" : "row-legit";
    const statusBadge = tx.is_fraud
      ? badge("FRAUD", "fraud")
      : badge("LEGIT", "legit");
    const riskBadge = badge(tx.risk, tx.risk.toLowerCase());
    return `
      <tr class="${rowCls}">
        <td class="text-slate-400">${tx.time}</td>
        <td class="font-semibold text-white">$${Number(tx.amt).toFixed(2)}</td>
        <td class="text-slate-300">${tx.category.replace(/_/g," ")}</td>
        <td class="text-slate-400">${fmt.num(tx.city_pop)}</td>
        <td>${riskBadge}</td>
        <td>${statusBadge}</td>
      </tr>`;
  }).join("");
}

// ── Live Polling ──────────────────────────────────────────────────────────────
function startPolling() {
  setInterval(async () => {
    await loadStats();
  }, 30000);
}

// ── Init history table ────────────────────────────────────────────────────────
renderHistoryTable();
