/**
 * dashboard.js — Sentra Fraud & AML Intelligence Platform
 * Handles Alert Investigation Workspace, Real-Data Network Graph,
 * SHAP Risk Explanations, Live Model Inference, and SQLite Decision Persistence.
 */

let currentAlertId = "ALT-10482";
let currentAlertData = null;
let allTransactions = [];

document.addEventListener("DOMContentLoaded", () => {
  loadAlertsList();
  loadTransactionsLedger();
  loadEvidenceData();

  // Close dropdowns on outside click
  document.addEventListener("click", (e) => {
    const btn = document.getElementById("actions-dropdown-btn");
    const menu = document.getElementById("actions-menu");
    if (btn && menu && !btn.contains(e.target) && !menu.contains(e.target)) {
      menu.classList.add("hidden");
    }
  });
});

// ── Tab Switching ──────────────────────────────────────────────────────────────
function switchTab(tabName) {
  const tabs = ["overview", "transactions", "connections", "evidence", "activity"];
  tabs.forEach((t) => {
    const content = document.getElementById(`tab-content-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (content) {
      if (t === tabName) {
        content.classList.remove("hidden");
      } else {
        content.classList.add("hidden");
      }
    }
    if (btn) {
      if (t === tabName) {
        btn.classList.add("active", "text-slate-900", "font-bold", "border-b-2", "border-blue-600");
        btn.classList.remove("text-slate-500", "font-medium", "border-transparent");
      } else {
        btn.classList.remove("active", "text-slate-900", "font-bold", "border-b-2", "border-blue-600");
        btn.classList.add("text-slate-500", "font-medium", "border-transparent");
      }
    }
  });

  if (tabName === "connections" && currentAlertData) {
    renderConnectionsGraph(currentAlertData);
  }
  if (tabName === "evidence") {
    loadEvidenceData();
  }
}

function switchCustTab(tab) {
  const custView = document.getElementById("cust-view-customer");
  const beneView = document.getElementById("cust-view-beneficiary");
  const custTab = document.getElementById("cust-tab-customer");
  const beneTab = document.getElementById("cust-tab-beneficiary");

  if (tab === "customer") {
    custView.classList.remove("hidden");
    beneView.classList.add("hidden");
    custTab.className = "text-xs font-bold text-slate-900 border-b-2 border-blue-600 pb-1";
    beneTab.className = "text-xs font-medium text-slate-400 hover:text-slate-700 pb-1";
  } else {
    custView.classList.add("hidden");
    beneView.classList.remove("hidden");
    beneTab.className = "text-xs font-bold text-slate-900 border-b-2 border-blue-600 pb-1";
    custTab.className = "text-xs font-medium text-slate-400 hover:text-slate-700 pb-1";
  }
}

function toggleActionsDropdown() {
  const menu = document.getElementById("actions-menu");
  if (menu) menu.classList.toggle("hidden");
}

// ── Load Alert Investigation Details ──────────────────────────────────────────
async function loadAlertDetails(alertId) {
  try {
    const res = await fetch(`/api/alerts/${alertId}`);
    if (!res.ok) throw new Error("Failed to load alert details");
    const data = await res.json();
    currentAlertData = data;
    renderAlertData(data);
  } catch (err) {
    console.error("Error loading alert:", err);
  }
}

function renderAlertData(data) {
  currentAlertId = data.id;

  // Header
  document.getElementById("header-alert-title").textContent = data.title || "Alert Investigation";
  document.getElementById("active-alert-pill").textContent = data.id;
  document.getElementById("header-alert-id").textContent = data.id;

  // Card 1: Metrics
  document.getElementById("card-flagged-amt").textContent = data.flagged_amount;
  document.getElementById("card-payment-type").textContent = data.payment_type;
  
  const riskScoreEl = document.getElementById("card-risk-score");
  riskScoreEl.textContent = `${data.risk_score}/100`;
  riskScoreEl.className = data.risk_score >= 70
    ? "text-3xl font-extrabold text-[#dc2626]"
    : (data.risk_score >= 35 ? "text-3xl font-extrabold text-[#d97706]" : "text-3xl font-extrabold text-[#16a34a]");
  
  document.getElementById("card-risk-level").textContent = data.risk_level;
  document.getElementById("card-detected-time").textContent = data.detected_time;
  document.getElementById("card-detected-reason").textContent = data.detected_reason;
  document.getElementById("card-sla-left").textContent = data.sla_left;
  document.getElementById("card-sla-due").textContent = data.sla_due;

  // Status Pill
  const statusPill = document.getElementById("card-status-pill");
  const statusText = document.getElementById("card-status-text");
  statusText.textContent = data.status;
  if (data.status === "Approved") {
    statusPill.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200";
  } else if (data.status === "Blocked / Frozen" || data.status === "Rejected") {
    statusPill.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-red-50 text-red-700 border border-red-200";
  } else if (data.status.includes("Escalated")) {
    statusPill.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200";
  } else {
    statusPill.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200";
  }

  // Timeline Nodes
  renderTimeline(data.timeline || []);

  // Card 2: Triggers (from SHAP explain_transaction output)
  renderTriggers(data.triggers || []);

  // Card 3: Connections
  if (data.explore_connections) {
    document.getElementById("connections-path-pill").textContent = data.explore_connections.path;
    document.getElementById("conn-headline").textContent = data.explore_connections.headline;
    document.getElementById("conn-desc").innerHTML = data.explore_connections.desc;
  }

  // Right Column: Customer Details (from real dataset fields)
  if (data.customer) {
    document.getElementById("cust-name").textContent = data.customer.name;
    document.getElementById("cust-id-badge").textContent = data.customer.id_badge;
    document.getElementById("cust-typical-amt").textContent = data.customer.typical_amount;
    document.getElementById("cust-largest-prior").textContent = data.customer.largest_prior;
    document.getElementById("cust-device-status").textContent = data.customer.session_device;
    document.getElementById("cust-payment-channel").textContent = data.customer.payment_channel;

    // Call modal data
    document.getElementById("call-modal-name").textContent = data.customer.name;
    document.getElementById("call-modal-phone").textContent = data.customer.phone || "+1 (555) 234-8921";
    document.getElementById("call-modal-email").textContent = data.customer.email || "client@example.com";
  }

  // Beneficiary / Merchant details
  if (data.beneficiary) {
    document.getElementById("bene-name").textContent = data.beneficiary.name;
    document.getElementById("bene-account").textContent = data.beneficiary.account_num;
    document.getElementById("bene-bank").textContent = data.beneficiary.bank;
    document.getElementById("bene-country").textContent = data.beneficiary.country;
    document.getElementById("bene-status").textContent = data.beneficiary.watchlist_status;
    document.getElementById("bene-dest").textContent = data.beneficiary.external_dest;
  }

  // AI Insight (from SHAP explain_transaction output)
  if (data.ai_insight) {
    document.getElementById("ai-badge").textContent = data.ai_insight.badge;
    document.getElementById("ai-title").textContent = data.ai_insight.title;
    document.getElementById("ai-desc").textContent = data.ai_insight.desc;
  }

  // Evidences text
  if (data.evidences_count) {
    document.getElementById("evidences-text").textContent = data.evidences_count;
  }

  // Render Connections Network Graph
  renderConnectionsGraph(data);
}

function renderTimeline(timeline) {
  const container = document.getElementById("timeline-nodes-container");
  if (!container || !timeline.length) return;

  container.innerHTML = timeline.map((node) => {
    const dotColor = node.dot_color || "green";
    const titleColor = node.title_color === "red" ? "text-[#ef4444]" : (node.title_color === "amber" ? "text-amber-600" : "text-slate-900");
    return `
      <div class="timeline-node">
        <div class="text-xs font-semibold text-slate-600 mb-1.5">${node.time}</div>
        <div class="timeline-circle ${dotColor}"></div>
        <div class="text-xs font-bold ${titleColor}">${node.title}</div>
        <div class="text-[11px] text-slate-500 mt-0.5">${node.account}</div>
        <div class="text-[11px] text-slate-400 mt-0.5">${node.subtext}</div>
      </div>
    `;
  }).join("");
}

function renderTriggers(triggers) {
  const container = document.getElementById("triggers-container");
  if (!container) return;

  container.innerHTML = triggers.map((trig, idx) => {
    const isLast = idx === triggers.length - 1;
    const borderClass = isLast ? "" : "pb-3.5 border-b border-slate-100";
    const tagClass = trig.tag_class || (trig.type === "Triggered rule" ? "badge-triggered" : "badge-signal");
    return `
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 ${borderClass}">
        <div>
          <div class="text-sm font-bold text-slate-900">${trig.title}</div>
          <div class="text-xs text-slate-500 mt-0.5">${trig.desc}</div>
        </div>
        <div class="flex items-center gap-3">
          <span class="${tagClass}">${trig.type}</span>
          <a href="#" onclick="switchTab('transactions'); return false;" class="text-xs text-blue-600 font-medium hover:underline">${trig.link_text || "View details"}</a>
        </div>
      </div>
    `;
  }).join("");
}

// ── Real Data Connections Network Graph ────────────────────────────────────────
function renderConnectionsGraph(alertData) {
  const container = document.getElementById("connections-svg-container");
  if (!container || !alertData) return;

  const conn = alertData.explore_connections || {};
  const merchantName = conn.merchant || alertData.beneficiary?.name || "Merchant";
  const city = conn.city || alertData.customer?.city || "Settlement Hub";
  const cardLast4 = conn.card_last4 || "Card";
  const customerName = alertData.customer?.name || "Cardholder";
  const amountStr = alertData.flagged_amount || "$0.00";
  const isHighRisk = alertData.risk_score >= 70;
  const sharedCards = conn.shared_cards || [];

  const merchantShort = merchantName.length > 18 ? merchantName.slice(0, 16) + "..." : merchantName;
  const cityShort = city.length > 18 ? city.slice(0, 16) + "..." : city;
  const cardBorder = isHighRisk ? "#ef4444" : "#10b981";
  const linkColor = isHighRisk ? "#ef4444" : "#3b82f6";

  const sharedCard1 = sharedCards[0] || "•0372";
  const sharedCard2 = sharedCards[1] || "•8308";

  container.innerHTML = `
    <svg class="w-full h-96" viewBox="0 0 800 360" fill="none">
      <!-- Background grid markers -->
      <circle cx="400" cy="180" r="160" stroke="#1e293b" stroke-width="1" stroke-dasharray="4 4" fill="none"/>
      
      <!-- Connecting lines -->
      <!-- Card -> Merchant -->
      <line x1="140" y1="180" x2="380" y2="180" stroke="${linkColor}" stroke-width="3" stroke-dasharray="6 6"/>
      
      <!-- Merchant -> City -->
      <line x1="420" y1="180" x2="660" y2="180" stroke="#3b82f6" stroke-width="2.5"/>
      
      <!-- Merchant -> Shared Card 1 (Top) -->
      <line x1="400" y1="150" x2="400" y2="70" stroke="#f59e0b" stroke-width="2"/>
      <line x1="400" y1="70" x2="620" y2="70" stroke="#ef4444" stroke-width="2" stroke-dasharray="4 4"/>

      <!-- Merchant -> Shared Card 2 (Bottom Right) -->
      <line x1="410" y1="210" x2="580" y2="290" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="4 4"/>

      <!-- Node 1: Originating Cardholder (Left) -->
      <circle cx="140" cy="180" r="34" fill="#1e293b" stroke="${cardBorder}" stroke-width="3"/>
      <text x="140" y="176" fill="#ffffff" font-size="12" font-weight="bold" text-anchor="middle">Card •${cardLast4}</text>
      <text x="140" y="192" fill="#94a3b8" font-size="10" text-anchor="middle">${customerName.slice(0, 12)}</text>
      <text x="140" y="234" fill="${cardBorder}" font-size="11" font-weight="600" text-anchor="middle">Flagged Origin</text>

      <!-- Amount pill between Card & Merchant -->
      <rect x="220" y="162" width="85" height="24" rx="5" fill="#0f172a" stroke="${linkColor}"/>
      <text x="262" y="178" fill="#60a5fa" font-size="11" font-weight="bold" text-anchor="middle">${amountStr}</text>

      <!-- Node 2: Central Merchant Hub (Center) -->
      <circle cx="400" cy="180" r="40" fill="#1e293b" stroke="#f59e0b" stroke-width="3.5"/>
      <text x="400" y="174" fill="#ffffff" font-size="11" font-weight="bold" text-anchor="middle">${merchantShort}</text>
      <text x="400" y="190" fill="#f59e0b" font-size="10" font-weight="bold" text-anchor="middle">Merchant Hub</text>
      <text x="400" y="240" fill="#f59e0b" font-size="11" font-weight="600" text-anchor="middle">Transaction Target</text>

      <!-- Edge label between Merchant and City -->
      <rect x="495" y="162" width="95" height="24" rx="5" fill="#0f172a" stroke="#3b82f6"/>
      <text x="542" y="178" fill="#93c5fd" font-size="10" font-weight="bold" text-anchor="middle">Settlement Location</text>

      <!-- Node 3: City Location (Right) -->
      <circle cx="660" cy="180" r="34" fill="#1e293b" stroke="#3b82f6" stroke-width="3"/>
      <text x="660" y="176" fill="#ffffff" font-size="11" font-weight="bold" text-anchor="middle">${cityShort}</text>
      <text x="660" y="192" fill="#94a3b8" font-size="10" text-anchor="middle">Local Cluster</text>
      <text x="660" y="234" fill="#3b82f6" font-size="11" font-weight="600" text-anchor="middle">Billing Region</text>

      <!-- Node 4: Co-occurring Card 1 (Top Right) -->
      <circle cx="620" cy="70" r="28" fill="#1e293b" stroke="#ef4444" stroke-width="2.5"/>
      <text x="620" y="73" fill="#ffffff" font-size="11" font-weight="bold" text-anchor="middle">${sharedCard1}</text>
      <text x="620" y="38" fill="#f87171" font-size="10" font-weight="bold" text-anchor="middle">Co-occurring Card</text>
      <text x="620" y="112" fill="#ef4444" font-size="9" text-anchor="middle">Shared Risky Merchant</text>

      <!-- Node 5: Co-occurring Card 2 (Bottom Right) -->
      <circle cx="580" cy="290" r="26" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
      <text x="580" y="294" fill="#ffffff" font-size="10" font-weight="bold" text-anchor="middle">${sharedCard2}</text>
      <text x="580" y="330" fill="#cbd5e1" font-size="9" text-anchor="middle">Linked Card (Same Terminal)</text>
    </svg>
  `;
}

// ── Evidence Tab Data Loader (from docs/results.json) ──────────────────────────
async function loadEvidenceData() {
  try {
    const res = await fetch("/api/evidence");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.models) return;

    // Populate XGBoost metrics
    const xgb = data.models.xgboost;
    if (xgb) {
      const opt = xgb.thresholds?.f1_optimal || {};
      const elPrauc = document.getElementById("ev-xgb-prauc");
      if (elPrauc) elPrauc.textContent = `${xgb.pr_auc.toFixed(4)}`;
      const elRoc = document.getElementById("ev-xgb-rocauc");
      if (elRoc) elRoc.textContent = `${xgb.roc_auc.toFixed(4)}`;
      const elRec = document.getElementById("ev-xgb-recall");
      if (elRec) elRec.textContent = `${(opt.recall * 100).toFixed(1)}%`;
      const elF1 = document.getElementById("ev-xgb-f1");
      if (elF1) elF1.textContent = `${opt.f1.toFixed(4)}`;
      const elSavings = document.getElementById("ev-xgb-savings");
      if (elSavings && opt.cost_analysis) {
        elSavings.textContent = `$${opt.cost_analysis.net_savings.toLocaleString()} (${opt.cost_analysis.savings_percentage}% saved)`;
      }
    }

    // Populate Ensemble metrics
    const ens = data.models.ensemble;
    if (ens) {
      const opt = ens.thresholds?.f1_optimal || {};
      const elPrauc = document.getElementById("ev-ens-prauc");
      if (elPrauc) elPrauc.textContent = `${ens.pr_auc.toFixed(4)}`;
      const elRoc = document.getElementById("ev-ens-rocauc");
      if (elRoc) elRoc.textContent = `${ens.roc_auc.toFixed(4)}`;
      const elPrec = document.getElementById("ev-ens-prec");
      if (elPrec) elPrec.textContent = `${(opt.precision * 100).toFixed(1)}%`;
      const elF1 = document.getElementById("ev-ens-f1");
      if (elF1) elF1.textContent = `${opt.f1.toFixed(4)}`;
      const elSavings = document.getElementById("ev-ens-savings");
      if (elSavings && opt.cost_analysis) {
        elSavings.textContent = `$${opt.cost_analysis.net_savings.toLocaleString()} (${opt.cost_analysis.savings_percentage}% saved)`;
      }
    }
  } catch (err) {
    console.error("Error loading evidence data:", err);
  }
}

// ── Alert Queue / Switcher ────────────────────────────────────────────────────
async function loadAlertsList() {
  try {
    const res = await fetch("/api/alerts");
    const data = await res.json();
    const container = document.getElementById("alerts-modal-list");
    if (!container) return;

    const alerts = data.alerts || [];
    container.innerHTML = alerts.map((alt) => {
      const riskClass = alt.risk_score >= 70 ? "text-red-600" : (alt.risk_score >= 35 ? "text-amber-600" : "text-emerald-600");
      return `
        <div onclick="openAlertCase('${alt.id}')" class="p-3.5 border border-slate-200 rounded-xl hover:border-blue-500 hover:bg-blue-50/30 cursor-pointer transition flex items-center justify-between">
          <div>
            <div class="flex items-center gap-2">
              <span class="font-bold text-slate-900 text-xs">${alt.id}</span>
              <span class="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">${alt.flagged_amount}</span>
              <span class="text-xs text-slate-500">• ${alt.customer_name}</span>
            </div>
            <div class="text-xs text-slate-500 mt-1">${alt.detected_reason} • ${alt.detected_time}</div>
          </div>
          <div class="text-right">
            <div class="text-sm font-extrabold ${riskClass}">${alt.risk_score}/100</div>
            <div class="text-[11px] text-slate-400 font-medium">${alt.status}</div>
          </div>
        </div>
      `;
    }).join("");

    // If initial alert is loaded, open the top-ranked alert
    if (alerts.length > 0 && (!currentAlertData || currentAlertId === "ALT-10482")) {
      loadAlertDetails(alerts[0].id);
    }
  } catch (e) {
    console.error("Error loading alerts list:", e);
  }
}

function openAlertCase(alertId) {
  closeAlertsModal();
  loadAlertDetails(alertId);
  switchTab("overview");
}

function openAlertsModal() {
  document.getElementById("modal-alerts-list").classList.remove("hidden");
}
function closeAlertsModal() {
  document.getElementById("modal-alerts-list").classList.add("hidden");
}

// ── Decision Modal & Submission (Persistent SQLite) ───────────────────────────
function openDecisionModal() {
  document.getElementById("modal-decision").classList.remove("hidden");
}
function closeDecisionModal() {
  document.getElementById("modal-decision").classList.add("hidden");
}

async function submitDecision(e) {
  e.preventDefault();
  const choice = document.querySelector('input[name="decision_choice"]:checked')?.value || "Approved";
  const notes = document.getElementById("decision-notes").value;

  try {
    const res = await fetch(`/api/alerts/${currentAlertId}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: choice, notes: notes, analyst: "Anna Reyes" }),
    });
    const result = await res.json();
    if (result.success) {
      closeDecisionModal();
      currentAlertData = result.alert;
      renderAlertData(result.alert);
      loadAlertsList(); // Refresh badge statuses in queue

      // Prepend to Activity tab
      const actContainer = document.getElementById("activity-log-container");
      if (actContainer) {
        const item = document.createElement("div");
        item.className = "flex items-start gap-3 pb-3 border-b border-slate-100 text-xs";
        const dotColor = choice === "Approved" ? "bg-emerald-500" : (choice === "Rejected" ? "bg-red-500" : "bg-amber-500");
        item.innerHTML = `
          <span class="w-2 h-2 rounded-full ${dotColor} mt-1.5 flex-shrink-0"></span>
          <div>
            <div class="font-bold text-slate-900">Analyst Decision: ${choice} (Saved to SQLite)</div>
            <div class="text-slate-500">${notes || `Alert manually marked as ${choice}.`}</div>
            <div class="text-[11px] text-slate-400 mt-1">Just now • Anna Reyes (Fraud Analyst L2)</div>
          </div>
        `;
        actContainer.prepend(item);
      }
    } else {
      alert("Error: " + (result.error || "Failed to record decision"));
    }
  } catch (err) {
    alert("Error recording decision: " + err.message);
  }
}

// ── Live ML Simulation Modal ──────────────────────────────────────────────────
function openLiveTestModal() {
  document.getElementById("modal-livetest").classList.remove("hidden");
}
function closeLiveTestModal() {
  document.getElementById("modal-livetest").classList.add("hidden");
}

async function runMLSimulation(e) {
  e.preventDefault();
  const btn = document.getElementById("sim-submit-btn");
  btn.disabled = true;
  btn.innerHTML = `<span>Evaluating model...</span>`;

  const payload = {
    amt: parseFloat(document.getElementById("sim-amt").value),
    category: document.getElementById("sim-category").value,
    hour: parseInt(document.getElementById("sim-hour").value),
    customer_name: document.getElementById("sim-customer").value,
    age: parseFloat(document.getElementById("sim-age").value),
    city_pop: parseFloat(document.getElementById("sim-pop").value),
    lat: 36.1627,
    long: -86.7816,
    gender: "F",
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const pred = await res.json();

    if (!res.ok) {
      const errDetails = Array.isArray(pred.details) ? pred.details.join("\n• ") : pred.error;
      alert(`Validation Error:\n• ${errDetails}`);
      return;
    }

    closeLiveTestModal();
    switchTab("overview");

    document.getElementById("card-flagged-amt").textContent = `$${payload.amt.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    document.getElementById("card-payment-type").textContent = `${payload.category.replace(/_/g, " ").toUpperCase()} • Instant ML Score`;
    
    const riskScoreEl = document.getElementById("card-risk-score");
    riskScoreEl.textContent = `${pred.risk_score}/100`;
    riskScoreEl.className = pred.risk_score >= 70
      ? "text-3xl font-extrabold text-[#dc2626]"
      : (pred.risk_score >= 35 ? "text-3xl font-extrabold text-[#d97706]" : "text-3xl font-extrabold text-[#16a34a]");

    document.getElementById("card-risk-level").textContent = pred.risk_level;
    document.getElementById("card-detected-reason").textContent = `Ensemble confidence: ${pred.confidence}% (XGBoost: ${pred.xgb_score}%, RF: ${pred.rf_score}%)`;

    // Render triggers & timeline from real prediction SHAP attributions
    if (pred.triggers) renderTriggers(pred.triggers);
    if (pred.timeline) renderTimeline(pred.timeline);
    if (pred.ai_insight) {
      document.getElementById("ai-badge").textContent = pred.ai_insight.badge;
      document.getElementById("ai-title").textContent = pred.ai_insight.title;
      document.getElementById("ai-desc").textContent = pred.ai_insight.desc;
    }
  } catch (err) {
    alert("Prediction error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>Run Model Scoring</span>`;
  }
}

// ── Call Customer Modal ───────────────────────────────────────────────────────
function openCustomerModal() {
  document.getElementById("modal-customer").classList.remove("hidden");
}
function closeCustomerModal() {
  document.getElementById("modal-customer").classList.add("hidden");
}

function simulateCall() {
  const btn = document.getElementById("btn-dial-sim");
  btn.disabled = true;
  btn.innerHTML = `<span>Connecting secure line...</span>`;
  btn.className = "w-full py-2.5 bg-amber-600 text-white font-bold rounded-lg transition shadow-sm flex items-center justify-center gap-2";

  setTimeout(() => {
    btn.innerHTML = `<span>Line Connected • Speaking to Cardholder</span>`;
    btn.className = "w-full py-2.5 bg-blue-600 text-white font-bold rounded-lg transition shadow-sm flex items-center justify-center gap-2";

    setTimeout(() => {
      btn.disabled = false;
      btn.innerHTML = `<span>Call Completed • Identity Confirmed</span>`;
      btn.className = "w-full py-2.5 bg-emerald-600 text-white font-bold rounded-lg transition shadow-sm flex items-center justify-center gap-2";
    }, 2500);
  }, 1800);
}

function openEvidenceModal() {
  switchTab("evidence");
}

function exportDossier() {
  alert(`Case Dossier for ${currentAlertId} exported successfully (JSON / Audit format).`);
}

// ── Transactions Ledger (from fraudTest.csv) ──────────────────────────────────
async function loadTransactionsLedger() {
  try {
    const res = await fetch("/api/transactions");
    const data = await res.json();
    allTransactions = data.transactions || [];
    renderTransactionsTable(allTransactions);
  } catch (e) {
    console.error("Error loading transactions:", e);
  }
}

function renderTransactionsTable(txList) {
  const tbody = document.getElementById("transactions-table-body");
  if (!tbody) return;

  if (!txList.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center py-8 text-slate-400">No transactions match your search.</td></tr>`;
    return;
  }

  tbody.innerHTML = txList.slice(0, 35).map((tx) => {
    const isFraud = tx.is_fraud === 1;
    const badgeHtml = isFraud
      ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-700">Fraud Flag</span>`
      : `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-700">Normal</span>`;
    const scoreColor = tx.risk_score >= 70 ? "text-red-600" : (tx.risk_score >= 35 ? "text-amber-600" : "text-emerald-600");

    return `
      <tr class="hover:bg-slate-50 transition">
        <td class="py-3 px-3 font-mono font-medium text-slate-800">${tx.trans_num.slice(0, 10)}...</td>
        <td class="py-3 px-3 text-slate-500">${tx.datetime}</td>
        <td class="py-3 px-3 font-semibold text-slate-900">${tx.customer}</td>
        <td class="py-3 px-3 text-slate-600">${tx.category}</td>
        <td class="py-3 px-3 text-slate-600">${tx.merchant}</td>
        <td class="py-3 px-3 font-bold text-slate-900">${tx.amount}</td>
        <td class="py-3 px-3 font-extrabold ${scoreColor}">${tx.risk_score}/100</td>
        <td class="py-3 px-3">${badgeHtml}</td>
        <td class="py-3 px-3 text-right">
          <button onclick="investigateTx('${tx.trans_num}', '${tx.customer}', ${tx.amt_raw}, '${tx.category}', ${tx.is_fraud})" class="text-blue-600 font-semibold hover:underline">
            Investigate →
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

function filterTransactionsTable() {
  const q = document.getElementById("tx-search-input").value.toLowerCase();
  const filtered = allTransactions.filter((tx) =>
    tx.customer.toLowerCase().includes(q) ||
    tx.merchant.toLowerCase().includes(q) ||
    tx.category.toLowerCase().includes(q) ||
    tx.trans_num.toLowerCase().includes(q)
  );
  renderTransactionsTable(filtered);
}

function investigateTx(id, name, amt, category, isFraud) {
  document.getElementById("sim-amt").value = amt.toFixed(2);
  document.getElementById("sim-customer").value = name;
  openLiveTestModal();
}
