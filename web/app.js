const $ = (id) => document.getElementById(id);

let userId = null;
let draftId = null;
let pollTimer = null;
const DATA_SEASON = 2026;

async function api(path) {
  const r = await fetch(path);
  const t = await r.text();
  let j;
  try {
    j = JSON.parse(t);
  } catch {
    throw new Error(t || r.statusText);
  }
  if (!r.ok) throw new Error(j.detail || j.message || t || r.statusText);
  return j;
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const t = await r.text();
  let j;
  try {
    j = JSON.parse(t);
  } catch {
    throw new Error(t || r.statusText);
  }
  if (!r.ok) throw new Error(j.detail || j.message || t || r.statusText);
  return j;
}

function setUserUi({ ok, displayLine, err }) {
  $("userOk").hidden = !ok;
  $("userError").hidden = !err;
  $("userDisplayLine").textContent = displayLine || "";
  $("userError").textContent = err || "";
}

$("btnUser").onclick = async () => {
  setUserUi({ ok: false, displayLine: "", err: "" });
  $("userError").hidden = false;
  $("userError").textContent = "Loading…";
  try {
    const u = $("username").value.trim();
    if (!u) throw new Error("Enter a username.");
    const j = await api(`/api/user?username=${encodeURIComponent(u)}`);
    userId = j.user_id;
    const uname = j.username || u;
    const dname = (j.display_name || "").trim();
    const line = dname ? `Signed in as @${uname} (${dname}).` : `Signed in as @${uname}.`;
    $("userError").hidden = true;
    setUserUi({ ok: true, displayLine: line, err: "" });
    $("btnDrafts").disabled = !userId;
  } catch (e) {
    userId = null;
    $("userOk").hidden = true;
    $("userError").hidden = false;
    $("userError").textContent = String(e.message || e);
    $("btnDrafts").disabled = true;
  }
};

$("btnDrafts").onclick = async () => {
  const sport = $("sport").value.trim() || "nfl";
  const season = $("season").value || "2026";
  $("draftList").innerHTML = "Loading…";
  try {
    const j = await api(`/api/drafts?user_id=${encodeURIComponent(userId)}&sport=${encodeURIComponent(sport)}&season=${season}`);
    $("draftList").innerHTML = "";
    if (!j.drafts || j.drafts.length === 0) {
      $("draftList").textContent = "No drafts returned (try another season).";
      return;
    }
    j.drafts.forEach((d) => {
      const row = document.createElement("div");
      row.className = "draft-row";
      const id = d.draft_id;
      const label = `${d.league_name || "League"} — ${d.status || "?"} — ${id}`;
      row.innerHTML = `<label><input type="radio" name="draftpick" value="${id}" /> ${escapeHtml(label)}</label>`;
      $("draftList").appendChild(row);
    });
    $("draftList").querySelectorAll('input[name="draftpick"]').forEach((el) => {
      el.addEventListener("change", () => {
        draftId = el.value;
        $("btnPoll").disabled = !draftId;
        $("btnRec").disabled = !draftId;
      });
    });
  } catch (e) {
    $("draftList").textContent = String(e.message || e);
  }
};

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderStatusHuman(j) {
  const el = $("statusHuman");
  const done = !j.in_progress;
  const yours = !!j.is_your_pick;
  const cur = j.current_team_idx;
  const mine = j.user_team_idx;
  let headline = "";
  if (done) headline = "Draft is finished (no more scheduled picks).";
  else if (yours) headline = "It is your pick — you can request recommendations.";
  else {
    const who = cur == null ? "another team" : `team slot ${cur}`;
    headline = `Waiting — ${who} is on the clock (you are slot ${mine}).`;
  }

  const sub = [
    `Sleeper draft status: ${j.draft_status || "?"}`,
    `Pick ${j.pick_no} of ${j.total_picks} (${j.picks_recorded} picks recorded).`,
  ].join(" ");

  el.innerHTML = `<p><strong>${escapeHtml(headline)}</strong></p><p class="muted">${escapeHtml(sub)}</p>`;
  el.hidden = false;
}

function buildPrefsFromUi() {
  const toNum = (id) => Number($(id).value);
  const risk = toNum("prefRisk");
  if (!Number.isFinite(risk) || risk < 0 || risk > 1) {
    throw new Error("Risk factor must be between 0 and 1.");
  }

  const ranks = {
    qb: toNum("rankQb"),
    rb: toNum("rankRb"),
    wr: toNum("rankWr"),
    te: toNum("rankTe"),
    k: toNum("rankK"),
    def: toNum("rankDef"),
  };

  const vals = Object.values(ranks);
  for (const v of vals) {
    if (!Number.isInteger(v) || v < 1 || v > 6) {
      throw new Error("Each position rank must be an integer from 1 to 6.");
    }
  }
  if (new Set(vals).size !== 6) {
    throw new Error("Position ranks must be unique (use each number 1..6 exactly once).");
  }

  const rankToWeight = (rank) => 7 - rank; // 1->6, 2->5, ... 6->1
  return {
    risk_factor: risk,
    w_qb: rankToWeight(ranks.qb),
    w_rb: rankToWeight(ranks.rb),
    w_wr: rankToWeight(ranks.wr),
    w_te: rankToWeight(ranks.te),
    w_k: rankToWeight(ranks.k),
    w_def: rankToWeight(ranks.def),
  };
}

async function pollOnce() {
  if (!draftId || !userId) return;
  const ds = DATA_SEASON;
  const q = `draft_id=${encodeURIComponent(draftId)}&user_id=${encodeURIComponent(userId)}&season=${encodeURIComponent(ds)}`;
  $("statusHuman").hidden = true;
  $("statusHuman").innerHTML = "<p class=\"muted\">Loading…</p>";
  $("statusHuman").hidden = false;
  try {
    const j = await api(`/api/draft-status?${q}`);
    renderStatusHuman(j);
    $("btnRec").disabled = false;
  } catch (e) {
    $("statusHuman").hidden = false;
    $("statusHuman").innerHTML = `<p class="muted">${escapeHtml(String(e.message || e))}</p>`;
  }
}

$("btnPoll").onclick = () => pollOnce();

$("autoPoll").onchange = () => {
  if ($("autoPoll").checked) {
    pollOnce();
    pollTimer = setInterval(pollOnce, 4000);
  } else if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
};

$("btnRec").onclick = async () => {
  if (!draftId || !userId) return;
  $("recTableWrap").hidden = false;
  $("recTableWrap").innerHTML = "<p>Loading…</p>";
  $("prefsError").hidden = true;
  $("prefsError").textContent = "";
  try {
    const prefs = buildPrefsFromUi();
    const body = {
      draft_id: draftId,
      user_id: userId,
      season: DATA_SEASON,
      candidates_k: 60,
      limit: 5,
      prefs,
    };
    const j = await apiPost("/api/recommend", body);
    const rows = j.recommendations || [];
    if (!rows.length) {
      $("recTableWrap").innerHTML = "<p>No recommendations returned.</p>";
    } else {
      const tr = rows
        .map((r, i) => {
          const name = escapeHtml(r.full_name || r.player_name || r.player_id || "");
          const pos = escapeHtml(r.pos || "");
          const team = escapeHtml(r.team || "");
          return `<tr>
            <td>${i + 1}</td>
            <td>${name}</td>
            <td>${pos}</td>
            <td>${team}</td>
          </tr>`;
        })
        .join("");
      $("recTableWrap").innerHTML = `<table class="rec-table">
        <thead><tr><th>#</th><th>Player</th><th>Pos</th><th>Team</th></tr></thead>
        <tbody>${tr}</tbody>
      </table>`;
    }
  } catch (e) {
    $("prefsError").hidden = false;
    $("prefsError").textContent = String(e.message || e);
    $("recTableWrap").innerHTML = `<p>${escapeHtml(String(e.message || e))}</p>`;
  }
};
