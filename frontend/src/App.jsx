import { useEffect, useMemo, useState } from "react";
import { getJson, postFile, postJson } from "./api";

const TABS = [
  { id: "monitor", label: "Live monitor" },
  { id: "analyze", label: "Analyze audio" },
  { id: "escalate", label: "Escalation" },
  { id: "reports", label: "Session report" },
];

export default function App() {
  const [tab, setTab] = useState("monitor");
  const [health, setHealth] = useState(null);
  const [session, setSession] = useState(null);
  const [reports, setReports] = useState([]);
  const [online, setOnline] = useState(false);
  const [seconds, setSeconds] = useState(134);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const h = await getJson("/api/v1/health");
        const s = await getJson("/api/v1/session");
        const r = await getJson("/api/v1/reports");
        if (!alive) return;
        setHealth(h);
        setSession(s);
        setReports(r);
        setOnline(true);
      } catch {
        if (!alive) return;
        setOnline(false);
        setSession(FALLBACK_SESSION);
        setReports(FALLBACK_REPORTS);
        setHealth(FALLBACK_HEALTH);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const id = setInterval(() => setSeconds((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const risk = session?.synthesis_risk?.risk_score_percentage ?? 87;
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <img className="brand-mark" src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=200&q=80" alt="" />
          <div>
            <p className="eyebrow">SIH 2026 · PS 26104 · Team TapuSena</p>
            <h1>
              HOMADOS <span>AI</span>
            </h1>
            <p className="sub">VoiceGuardAI · voice-cloning fraud detection</p>
          </div>
        </div>
        <div className="top-meta">
          <span className={`pill ${online ? "ok" : "warn"}`}>
            {online ? "Public API connected" : "Demo mode · API unreachable"}
          </span>
          <div className="agent">
            <img src="https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=200&q=80" alt="Agent Priya Sharma" />
            <div>
              <strong>Priya Sharma</strong>
              <span>Mumbai Central · Fraud Ops</span>
            </div>
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Dashboard">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={tab === item.id ? "active" : ""}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "monitor" && (
          <Monitor
            session={session}
            health={health}
            risk={risk}
            clock={`${mm}:${ss}`}
            onEscalate={() => setTab("escalate")}
          />
        )}
        {tab === "analyze" && <Analyze health={health} />}
        {tab === "escalate" && <Escalate />}
        {tab === "reports" && <Reports rows={reports} />}
      </main>

      <footer>
        <span>Zero-retention · feature vectors only · no raw audio stored</span>
        <span>ECAPA-TDNN pretrained attached · training deferred to HPC</span>
      </footer>
    </div>
  );
}

function Monitor({ session, health, risk, clock, onEscalate }) {
  const caller = session?.claimed_caller;
  return (
    <section className="stack">
      <div className="hero-card">
        <img className="hero-bg" src="https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=1400&q=80" alt="" />
        <div className="hero-copy">
          <p className="eyebrow">Inbound telephony stream</p>
          <h2>{session?.incoming_number || "+91 98XXX XX210"}</h2>
          <p>
            Claimed identity <strong>{caller?.name || "Rajesh Kumar"}</strong> ·{" "}
            {session?.transaction_context?.type} {session?.transaction_context?.amount_inr}
          </p>
        </div>
        <div className="live">
          <span className="dot" /> Live {clock}
        </div>
      </div>

      <div className="grid-2">
        <article className="card">
          <div className="card-head">
            <div className="who">
              <img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=200&q=80" alt="Claimed caller" />
              <div>
                <p className="eyebrow">KYC claimed caller</p>
                <h3>{caller?.name}</h3>
                <p>
                  {caller?.account_type} · {caller?.account_number_masked}
                </p>
              </div>
            </div>
            <span className="flag">VoIP · unregistered SIM</span>
          </div>
          <Waveform />
          <Spectrogram />
          <div className="bio-row">
            <div>
              <span>Enrolled voiceprint</span>
              <div className="mini-bars baseline">{bars(8, 0.2)}</div>
            </div>
            <b>VS</b>
            <div>
              <span>Live inbound stream</span>
              <div className="mini-bars live">{bars(8, 0.7)}</div>
            </div>
          </div>
        </article>

        <article className="card gauge-card">
          <p className="eyebrow">Synthesis probability</p>
          <h3>Voice clone risk</h3>
          <Gauge value={risk} />
          <p className="verdict">
            {session?.synthesis_risk?.risk_label || "High risk likely synthetic voice"}
          </p>
          <ul className="model-list">
            <li>Spectral / mel / MFCC / pYIN · live</li>
            <li>
              ECAPA-TDNN · {health?.models?.ecapa_tdnn?.inference_ready ? "pretrained ready" : "attached, weights downloadable"}
            </li>
            <li>
              AASIST · {health?.models?.aasist?.inference_ready ? "checkpoint loaded" : "repo wired, spectral prior until weights"}
            </li>
          </ul>
          <button className="primary" onClick={onEscalate}>
            Recommend callback verification
          </button>
        </article>
      </div>

      <article className="card">
        <p className="eyebrow">Explainability</p>
        <h3>Detected artifacts</h3>
        <div className="art-grid">
          {(FALLBACK_TELEMETRY).map((item) => (
            <div key={item.name} className="art">
              <div className="art-top">
                <span>{item.type}</span>
                <strong>{item.score}</strong>
              </div>
              <h4>{item.name}</h4>
              <p>{item.description}</p>
              <div className="meter">
                <i style={{ width: `${Math.min(100, item.score * (item.score > 1 ? 1 : 100))}%` }} />
              </div>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}

function Analyze({ health }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(e) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const json = await postFile("/api/v1/analyze", file);
      setResult(json);
    } catch (err) {
      setError(err.message || "Analysis failed. Start the API or set VITE_API_URL.");
    } finally {
      setBusy(false);
    }
  }

  const fusion = result?.fusion;
  const preview = result?.spectral?.spectrogram_preview || [];

  return (
    <section className="stack">
      <article className="card">
        <p className="eyebrow">Any network · no localhost lock-in</p>
        <h2>Upload a call clip</h2>
        <p className="lede">
          Runs spectral analysis immediately. ECAPA-TDNN embeddings load from the SpeechBrain
          VoxCeleb checkpoint when installed; training/fine-tune is reserved for supercomputer time.
        </p>
        <form className="upload" onSubmit={run}>
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.ogg"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button className="primary" disabled={!file || busy}>
            {busy ? "Scoring…" : "Run fusion engine"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
        <p className="hint">
          Models: spectral {health?.models?.spectral?.inference_ready ? "ready" : "—"}. ECAPA{" "}
          {health?.models?.ecapa_tdnn?.detail}. AASIST {health?.models?.aasist?.detail}.
        </p>
      </article>

      {fusion && (
        <div className="grid-2">
          <article className="card">
            <Gauge value={fusion.risk_score_percentage} />
            <p className="verdict">{fusion.risk_label}</p>
            <dl className="kv">
              <div>
                <dt>Spoof probability</dt>
                <dd>{fusion.spoof_probability}</dd>
              </div>
              <div>
                <dt>Identity cosine</dt>
                <dd>{fusion.identity_cosine ?? "no enrollment"}</dd>
              </div>
              <div>
                <dt>Speaker backend</dt>
                <dd>{result.speaker.backend}</dd>
              </div>
              <div>
                <dt>Duration</dt>
                <dd>{result.spectral.duration_sec}s</dd>
              </div>
            </dl>
          </article>
          <article className="card">
            <h3>Log-mel preview</h3>
            <Heatmap matrix={preview} />
            <div className="art-grid compact">
              {result.artifacts?.map((item) => (
                <div key={item.name} className="art">
                  <div className="art-top">
                    <span>{item.type}</span>
                    <strong>{item.score}</strong>
                  </div>
                  <h4>{item.name}</h4>
                </div>
              ))}
            </div>
          </article>
        </div>
      )}
    </section>
  );
}

function Escalate() {
  const [toast, setToast] = useState("");

  async function act(action) {
    try {
      const data = await postJson("/api/v1/escalate", { action, session_id: "#HOM-2026-9810" });
      setToast(data.message);
    } catch {
      setToast("Escalation recorded locally. Connect the public API to dispatch for real.");
    }
  }

  return (
    <section className="stack">
      <div className="alert-banner">
        <img src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1400&q=80" alt="" />
        <div>
          <p className="eyebrow">Level-3 fraud intervention</p>
          <h2>Call flagged for secondary verification</h2>
          <p>₹4,50,000 RTGS must not proceed until the claimed customer is verified out-of-band.</p>
        </div>
      </div>
      <div className="cards-3">
        <EscalationCard
          img="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=800&q=80"
          title="Call back on registered number"
          body="Drop the VoIP leg and place a PSTN call to the KYC SIM."
          action={() => act("callback")}
          cta="Initiate verified callback"
          tag="Recommended"
        />
        <EscalationCard
          img="https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=800&q=80"
          title="Trigger OTP to registered device"
          body="Push a step-up challenge to the enrolled mobile banking app."
          action={() => act("otp")}
          cta="Send in-app challenge"
          tag="Step-up"
        />
        <EscalationCard
          img="https://images.unsplash.com/photo-1560250097-0b93528c311a?auto=format&fit=crop&w=800&q=80"
          title="Escalate to supervisor"
          body="Hand fusion telemetry to the senior cybercrime desk."
          action={() => act("supervisor")}
          cta="Transfer to senior desk"
          tag="Investigation"
        />
      </div>
      {toast && <div className="toast">{toast}</div>}
    </section>
  );
}

function EscalationCard({ img, title, body, action, cta, tag }) {
  return (
    <article className="card escalate-card">
      <div className="photo">
        <img src={img} alt="" />
        <span>{tag}</span>
      </div>
      <h3>{title}</h3>
      <p>{body}</p>
      <button className="primary" onClick={action}>
        {cta}
      </button>
    </article>
  );
}

function Reports({ rows }) {
  const data = rows?.length ? rows : FALLBACK_REPORTS;
  return (
    <section className="stack">
      <div className="kpis">
        <div className="card kpi">
          <span>Calls screened</span>
          <strong>142</strong>
        </div>
        <div className="card kpi">
          <span>Clones intercepted</span>
          <strong>3</strong>
        </div>
        <div className="card kpi">
          <span>Loss prevented</span>
          <strong>₹18.4L</strong>
        </div>
        <div className="card kpi">
          <span>Avg verify time</span>
          <strong>1.8s</strong>
        </div>
      </div>
      <article className="card table-wrap">
        <h3>Audit log</h3>
        <table>
          <thead>
            <tr>
              <th>Session</th>
              <th>Risk</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.session_id} className={row.status}>
                <td>
                  <strong>{row.timestamp}</strong>
                  <div>{row.session_id}</div>
                </td>
                <td>
                  {row.risk_score}% · {row.risk_label}
                </td>
                <td>
                  {row.action_taken}
                  <div>{row.action_sub}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>
    </section>
  );
}

function Gauge({ value = 0 }) {
  const offset = 528 - (528 * Math.min(100, value)) / 100;
  return (
    <div className="gauge">
      <svg viewBox="0 0 200 200">
        <circle cx="100" cy="100" r="84" className="track" />
        <circle
          cx="100"
          cy="100"
          r="84"
          className="progress"
          strokeDasharray="528"
          strokeDashoffset={offset}
        />
      </svg>
      <div>
        <b>{Math.round(value)}</b>
        <span>% risk</span>
      </div>
    </div>
  );
}

function Waveform() {
  const items = useMemo(() => Array.from({ length: 48 }, (_, i) => i), []);
  return (
    <div className="wave" aria-hidden="true">
      {items.map((i) => (
        <span key={i} className={i > 28 && i < 36 ? "hot" : ""} style={{ animationDelay: `${i * 40}ms` }} />
      ))}
    </div>
  );
}

function Spectrogram() {
  return (
    <div className="spec">
      <div className="spec-label">
        <span>0 Hz</span>
        <span>3.2 kHz vocoder band</span>
        <span>8 kHz</span>
      </div>
      <div className="spec-bar" />
    </div>
  );
}

function Heatmap({ matrix }) {
  if (!matrix?.length) return null;
  return (
    <div className="heat" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)` }}>
      {matrix.flatMap((row, i) =>
        row.map((v, j) => (
          <i key={`${i}-${j}`} style={{ opacity: 0.25 + v * 0.75 }} />
        ))
      )}
    </div>
  );
}

function bars(n, seed) {
  return Array.from({ length: n }, (_, i) => (
    <i key={i} style={{ height: `${30 + ((i * 17 + seed * 50) % 60)}%` }} />
  ));
}

const FALLBACK_HEALTH = {
  models: {
    ecapa_tdnn: { inference_ready: false, detail: "pretrained download pending" },
    aasist: { inference_ready: false, detail: "spectral prior" },
    spectral: { inference_ready: true },
  },
};

const FALLBACK_SESSION = {
  incoming_number: "+91 98XXX XX210",
  claimed_caller: {
    name: "Rajesh Kumar",
    account_type: "Priority Banking",
    account_number_masked: "••••4912",
  },
  transaction_context: { type: "Urgent RTGS", amount_inr: "₹4,50,000" },
  synthesis_risk: { risk_score_percentage: 87, risk_label: "High risk likely synthetic voice" },
};

const FALLBACK_REPORTS = [
  {
    session_id: "#HOM-2026-9810",
    timestamp: "10:14:02 IST",
    risk_score: 87,
    risk_label: "Critical (Synthetic Voice)",
    action_taken: "Callback Verification Recommended",
    action_sub: "VoIP spoof flagged",
    status: "danger",
  },
  {
    session_id: "#HOM-2026-9809",
    timestamp: "09:48:19 IST",
    risk_score: 14,
    risk_label: "Safe (Genuine Human)",
    action_taken: "Voice biometrics cleared",
    action_sub: "Matches enrolled profile",
    status: "safe",
  },
  {
    session_id: "#HOM-2026-9808",
    timestamp: "09:12:45 IST",
    risk_score: 62,
    risk_label: "Suspicious (Elevated Jitter)",
    action_taken: "OTP step-up dispatched",
    action_sub: "Push auth passed",
    status: "warning",
  },
];

const FALLBACK_TELEMETRY = [
  {
    type: "Acoustic",
    name: "Spectral discontinuity at 3.2 kHz",
    score: 0.94,
    description: "High-band cutoff typical of neural vocoders.",
  },
  {
    type: "Prosody",
    name: "Unnatural pitch flattening",
    score: 0.91,
    description: "Missing micro-tremor in the F0 contour.",
  },
  {
    type: "Vocoder",
    name: "HiFi-GAN / DiffSinger traces",
    score: 0.88,
    description: "Checkerboard phase noise in transition frames.",
  },
  {
    type: "Identity",
    name: "Voiceprint divergence 43.8%",
    score: 0.438,
    description: "ECAPA cosine drift vs enrolled customer vector.",
  },
];
