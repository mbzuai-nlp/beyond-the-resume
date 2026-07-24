import { render } from "preact";
import { useEffect, useState } from "preact/hooks";
import "./styles.css";

const LEVELS = ["low", "medium", "high"];
const LEVEL_LABELS = { low: "L", medium: "M", high: "H" };

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatDuration(milliseconds) {
  if (milliseconds < 1) return "<1 ms";
  if (milliseconds < 1000) return `${milliseconds.toFixed(0)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${(milliseconds / 60000).toFixed(1)} min`;
}

function Sidebar({ username, rubricOpen, onDashboard, onRubric, onSignOut }) {
  return (
    <aside class="sidebar">
      <div class="brand">
        <img class="brand-mark" src="logo_btr.png" alt="" />
        <span>Beyond the Résumé</span>
      </div>

      <nav aria-label="Console navigation">
        <button
          class={`nav-item${rubricOpen ? "" : " active"}`}
          onClick={onDashboard}
        >
          <span class="nav-icon">⌂</span>
          Dashboard
        </button>
        <button
          class={`nav-item${rubricOpen ? " active" : ""}`}
          onClick={onRubric}
        >
          <span class="nav-icon">≡</span>
          Rubric
        </button>
      </nav>

      <div class="reviewer-account">
        <span>Signed in as</span>
        <strong>{username || "Reviewer"}</strong>
        <button onClick={onSignOut}>Sign out</button>
      </div>
    </aside>
  );
}

function EmptyState() {
  return (
    <div class="empty-state">
      <span>◎</span>
      <h2>No applications yet</h2>
      <p>Submitted interviews will appear here.</p>
    </div>
  );
}

function ApplicationsTable({ applications, onSelect }) {
  if (!applications.length) return <EmptyState />;

  return (
    <div class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Applicant</th>
            <th>Résumé</th>
            <th>Submitted</th>
            <th>Messages</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {applications.map((application) => (
            <tr
              key={application.email}
              onClick={() => onSelect(application.email)}
            >
              <td>
                <strong>{application.email}</strong>
              </td>
              <td>{application.resume_filename}</td>
              <td>{formatDate(application.created_at)}</td>
              <td>{application.message_count}</td>
              <td class="row-arrow">→</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Dashboard({ applications, loading, error, onSelect }) {
  return (
    <section class="page">
      <header class="page-header">
        <div>
          <p class="eyebrow">Overview</p>
          <h1>Applications</h1>
          <p>Review applicant résumés and interview transcripts.</p>
        </div>
        <span class="application-count">
          {applications.length} {applications.length === 1 ? "application" : "applications"}
        </span>
      </header>

      {loading && <div class="loading-state">Loading applications…</div>}
      {error && <div class="error-state">{error}</div>}
      {!loading && !error && (
        <ApplicationsTable applications={applications} onSelect={onSelect} />
      )}
    </section>
  );
}

function RubricModal({ rubric, loading, error, onClose }) {
  return (
    <div class="rubric-modal-backdrop" onClick={onClose}>
      <section
        class="rubric-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rubric-title"
        onClick={(event) => event.stopPropagation()}
      >
      <header class="rubric-modal-header">
        <div>
          <p class="eyebrow">Interview criteria</p>
          <h1 id="rubric-title">Rubric</h1>
          <p>{rubric?.description || "Loading the interview rubric."}</p>
        </div>
        <button onClick={onClose} aria-label="Close rubric">×</button>
      </header>

        <div class="rubric-modal-body">
          {loading && <div class="loading-state">Loading rubric…</div>}
          {error && <div class="error-state">{error}</div>}
          {rubric && !loading && !error && (
            <div class="rubric-list">
              {rubric.dimensions.map((dimension, index) => (
                <section class="dimension-card" key={dimension.id}>
                  <header>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <h2>{dimension.name}</h2>
                  </header>
                  <div class="level-grid">
                    {LEVELS.map((level) => (
                      <article class={`level ${level}`} key={level}>
                        <strong>{level}</strong>
                        <p>{dimension.levels[level]}</p>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function messageTiming(message, previousMessage) {
  if (
    message.role === "interviewer" &&
    message.generation_started_at &&
    message.generation_completed_at
  ) {
    return `Generated in ${formatDuration(
      new Date(message.generation_completed_at) -
        new Date(message.generation_started_at),
    )}`;
  }

  if (message.role === "applicant" && previousMessage) {
    return `Replied after ${formatDuration(
      new Date(message.created_at) - new Date(previousMessage.created_at),
    )}`;
  }
}

function BeliefBarChart({ posteriors }) {
  return (
    <div class="belief-bar-chart">
      {LEVELS.map((level) => {
        const value = posteriors[level];
        return (
          <div class={`belief-bar-column ${level}`} key={level}>
            <div class="belief-bar-axis">
              <i style={{ height: `${value * 100}%` }} />
            </div>
            <span>
              <strong>{Math.round(value * 100)}%</strong>
              {LEVEL_LABELS[level]}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function formatPointChange(value) {
  const points = Math.round(value * 100);
  return `${points > 0 ? "+" : ""}${points}pp`;
}

function BeliefDeltaChart({ previous, current }) {
  return (
    <div class="belief-bar-chart delta-chart">
      {LEVELS.map((level) => {
        const before = previous[level];
        const after = current[level];
        const change = after - before;
        const direction =
          change > 0 ? "increase" : change < 0 ? "decrease" : "stable";

        return (
          <div class={`belief-bar-column ${direction}`} key={level}>
            <div class="belief-bar-axis">
              <i class="original" style={{ height: `${before * 100}%` }} />
              {change !== 0 && (
                <b
                  class="delta"
                  style={{
                    bottom: `${Math.min(before, after) * 100}%`,
                    height: `${Math.abs(change) * 100}%`,
                  }}
                />
              )}
              <em style={{ bottom: `${after * 100}%` }} />
            </div>
            <span>
              <strong class={direction}>{formatPointChange(change)}</strong>
              {LEVEL_LABELS[level]}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function BeliefPopover({ changes }) {
  return (
    <aside class="belief-popover">
      <strong>Belief changes</strong>
      {changes.length ? (
        changes.map((change) => (
          <article key={change.id}>
            <BeliefDeltaChart
              previous={change.previous}
              current={change.current}
            />
            <div class="belief-change-copy">
              <h4>{change.name}</h4>
              <p>{change.justification}</p>
            </div>
          </article>
        ))
      ) : (
        <p>No distribution changed by at least one percentage point.</p>
      )}
    </aside>
  );
}

function ResumeBeliefPopover({ update, rubric }) {
  return (
    <aside class="belief-popover resume-belief-popover">
      <strong>Résumé belief</strong>
      {rubric.dimensions.map((dimension) => (
        <article key={dimension.id}>
          <BeliefBarChart posteriors={update.posteriors[dimension.id]} />
          <div class="belief-change-copy">
            <h4>{dimension.name}</h4>
            <p>{update.justifications[dimension.id]}</p>
          </div>
        </article>
      ))}
    </aside>
  );
}

function FinalBelief({ update, rubric }) {
  return (
    <section class="final-belief">
      <header>
        <strong>Final belief</strong>
        <time>{formatDate(update.created_at)}</time>
      </header>
      <div>
        {rubric.dimensions.map((dimension) => (
          <article key={dimension.id}>
            <BeliefBarChart posteriors={update.posteriors[dimension.id]} />
            <h3>{dimension.name}</h3>
          </article>
        ))}
      </div>
    </section>
  );
}

function beliefChangesByMessage(updates, rubric) {
  const changes = {};
  let previous;

  updates.forEach((update) => {
    if (update.applicant_message_id === null) {
      previous = update;
      return;
    }

    if (previous) {
      changes[update.applicant_message_id] = rubric.dimensions
        .filter((dimension) =>
          LEVELS.some(
            (level) =>
              update.posteriors[dimension.id][level] -
                previous.posteriors[dimension.id][level] >=
              0.01,
          ),
        )
        .map((dimension) => ({
          id: dimension.id,
          name: dimension.name,
          previous: previous.posteriors[dimension.id],
          current: update.posteriors[dimension.id],
          justification: update.justifications[dimension.id],
        }));
    }
    previous = update;
  });

  return changes;
}

function Transcript({ messages, beliefUpdates, rubric, uploadedAt }) {
  const beliefChanges = beliefChangesByMessage(beliefUpdates, rubric);
  const resumeBelief = beliefUpdates.find(
    (update) => update.applicant_message_id === null,
  );
  const finalBelief = [...beliefUpdates]
    .reverse()
    .find((update) => update.applicant_message_id !== null);

  return (
    <div class="transcript">
      <article
        class={`transcript-message applicant resume-upload${
          resumeBelief ? " has-belief" : ""
        }`}
      >
        <header>
          <strong>
            Applicant
            {resumeBelief && <span class="belief-indicator">Belief</span>}
          </strong>
          <time>{formatDate(uploadedAt)}</time>
        </header>
        <p>Uploaded résumé</p>
        {resumeBelief && (
          <ResumeBeliefPopover update={resumeBelief} rubric={rubric} />
        )}
      </article>
      {messages.map((message, index) => {
        const timing = messageTiming(message, messages[index - 1]);
        const changes = beliefChanges[message.id];

        return (
          <article
            class={`transcript-message ${message.role}${
              changes ? " has-belief" : ""
            }`}
            key={message.id}
          >
            <header>
              <strong>
                {message.role === "interviewer" && (
                  <img class="message-logo" src="logo_btr.png" alt="" />
                )}
                {message.role === "interviewer" ? "Interviewer" : "Applicant"}
                {changes && <span class="belief-indicator">Belief</span>}
              </strong>
              <time>{formatDate(message.created_at)}</time>
            </header>
            <p>{message.content}</p>
            {timing && <small>{timing}</small>}
            {changes && <BeliefPopover changes={changes} />}
          </article>
        );
      })}
      {finalBelief && <FinalBelief update={finalBelief} rubric={rubric} />}
    </div>
  );
}

function ReviewPanel({
  messages,
  beliefUpdates,
  rubric,
  uploadedAt,
  onCollapse,
}) {
  return (
    <section class="transcript-panel">
      <header class="panel-header review-panel-header">
        <strong class="review-title">
          Transcript <small>{messages.length + 1}</small>
        </strong>
        <button onClick={onCollapse} aria-label="Collapse review panel">×</button>
      </header>
      <Transcript
        messages={messages}
        beliefUpdates={beliefUpdates}
        rubric={rubric}
        uploadedAt={uploadedAt}
      />
    </section>
  );
}

function ApplicationDetail({ detail, onBack }) {
  const [panel, setPanel] = useState("transcript");
  const { application, messages, belief_updates: beliefUpdates, rubric } = detail;
  const encodedEmail = encodeURIComponent(application.email);
  const resumeUrl = `api/reviewer/applications/${encodedEmail}/resume`;

  return (
    <section class="detail-page">
      <header class="detail-header">
        <button class="back-button" onClick={onBack}>← Applications</button>
        <div>
          <h1>{application.email}</h1>
          <p>
            Submitted {formatDate(application.created_at)}
            <span>·</span>
            {messages.length} messages
          </p>
        </div>
        <a href={resumeUrl} target="_blank" rel="noreferrer">Open PDF ↗</a>
      </header>

      <div class={`review-workspace${panel ? " transcript-open" : ""}`}>
        <section class="resume-panel">
          <header class="panel-header">
            <div>
              <span>Résumé</span>
              <strong>{application.resume_filename}</strong>
            </div>
          </header>
          <iframe src={resumeUrl} title={`${application.email} résumé`} />
        </section>

        {panel ? (
          <ReviewPanel
            messages={messages}
            beliefUpdates={beliefUpdates}
            rubric={rubric}
            uploadedAt={application.created_at}
            onCollapse={() => setPanel()}
          />
        ) : (
          <button
            class="transcript-toggle"
            onClick={() => setPanel("transcript")}
          >
            <span>→</span>
            View transcript
            <small>{messages.length}</small>
          </button>
        )}
      </div>
    </section>
  );
}

function App() {
  const [username, setUsername] = useState("");
  const [applications, setApplications] = useState([]);
  const [rubric, setRubric] = useState();
  const [rubricOpen, setRubricOpen] = useState(false);
  const [rubricLoading, setRubricLoading] = useState(false);
  const [rubricError, setRubricError] = useState("");
  const [selected, setSelected] = useState();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("api/session")
      .then((response) => response.json())
      .then((session) => setUsername(session.username));
    loadApplications();
  }, []);

  async function loadApplications() {
    setRubricOpen(false);
    setSelected();
    setLoading(true);
    setError("");

    try {
      const response = await fetch("api/reviewer/applications");
      if (!response.ok) throw new Error();
      setApplications(await response.json());
    } catch {
      setError("Applications could not be loaded.");
    } finally {
      setLoading(false);
    }
  }

  async function loadRubric() {
    setRubricOpen(true);
    if (rubric) return;
    setRubricLoading(true);
    setRubricError("");

    try {
      const response = await fetch("api/reviewer/rubric");
      if (!response.ok) throw new Error();
      setRubric(await response.json());
    } catch {
      setRubricError("The rubric could not be loaded.");
    } finally {
      setRubricLoading(false);
    }
  }

  async function openApplication(email) {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `api/reviewer/applications/${encodeURIComponent(email)}`,
      );
      if (!response.ok) throw new Error();
      setSelected(await response.json());
    } catch {
      setError("The application could not be loaded.");
    } finally {
      setLoading(false);
    }
  }

  function signOut() {
    const request = new XMLHttpRequest();
    request.open("GET", "api/session", true, "signed-out", String(Date.now()));
    request.onloadend = () => window.location.reload();
    request.send();
  }

  return (
    <div class="console">
      <Sidebar
        username={username}
        rubricOpen={rubricOpen}
        onDashboard={loadApplications}
        onRubric={loadRubric}
        onSignOut={signOut}
      />
      <main class="content">
        {selected ? (
          <ApplicationDetail detail={selected} onBack={loadApplications} />
        ) : (
          <Dashboard
            applications={applications}
            loading={loading}
            error={error}
            onSelect={openApplication}
          />
        )}
      </main>
      {rubricOpen && (
        <RubricModal
          rubric={rubric}
          loading={rubricLoading}
          error={rubricError}
          onClose={() => setRubricOpen(false)}
        />
      )}
    </div>
  );
}

render(<App />, document.querySelector("#app"));
