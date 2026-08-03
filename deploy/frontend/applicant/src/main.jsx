import { render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import "./styles.css";

function Progress({ step }) {
  return (
    <div class="progress" aria-label={`Step ${step} of 3`}>
      <span>{step} of 3</span>
      <div class="progress-track">
        <div class="progress-value" style={{ width: `${(step / 3) * 100}%` }} />
      </div>
    </div>
  );
}

function ConsentStep({
  privacyConsent,
  setPrivacyConsent,
  consent,
  setConsent,
  reviewConsent,
  setReviewConsent,
  onContinue,
}) {
  return (
    <section class="setup-card">
      <Progress step={1} />
      <div class="setup-copy">
        <p class="eyebrow">ML Engineer Interview</p>
        <h1>Welcome to your interview.</h1>
        <p class="lead">This interview will consist of 10 questions guided by an AI. The interview context is for a generic Machine Learning Engineering position that values technical, engineering, and soft skills.</p>
      </div>
      <form class="setup-form" onSubmit={onContinue}>
        <div class="consent-list">
          <label class="consent-field">
            <input
              type="checkbox"
              checked={privacyConsent}
              onChange={(event) =>
                setPrivacyConsent(event.currentTarget.checked)
              }
              required
            />
            <span>
              I agree not to provide any private information to this system and
              will only upload a resume that contains information I would happily
              share in a public forum.
            </span>
          </label>
          <label class="consent-field">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.currentTarget.checked)}
              required
            />
            <span>
              I consent to aggregate metrics derived from my interaction and
              provided data being published in a research publication. My
              résumé, interview transcript, logs, other granular data, and
              personally identifiable information will NOT be published or 
              shared.
            </span>
          </label>
          <label class="consent-field">
            <input
              type="checkbox"
              checked={reviewConsent}
              onChange={(event) => setReviewConsent(event.currentTarget.checked)}
              required
            />
            <span>
              I consent to an authorized human reviewer reading my résumé and
              interview transcript for the purpose of computing the aggregate
              research statistics described above.
            </span>
          </label>
        </div>
        <button class="primary-button" type="submit">
          Continue
          <span aria-hidden="true">→</span>
        </button>
      </form>
    </section>
  );
}

function ResumeProcessing() {
  return (
    <section class="setup-card processing-card" role="status" aria-live="polite">
      <div class="processing-mark">
        <img src="logo_btr.png" alt="" />
      </div>
      <p class="eyebrow">Preparing your interview</p>
      <h1>Reading your résumé.</h1>
      <p class="lead">
        We’re preparing the first interview question. This may take a minute or
        so.
      </p>
    </section>
  );
}

function ResumeStep({
  resume,
  setResume,
  starting,
  error,
  onBack,
  onContinue,
}) {
  return (
    <section class="setup-card">
      <Progress step={2} />
      <div class="setup-copy">
        <p class="eyebrow">Your background</p>
        <h1>Add your résumé.</h1>
        <p class="lead">
          Upload the résumé you’d like the interviewer to reference.
        </p>
      </div>
      <form class="setup-form" onSubmit={onContinue}>
        <label class={`upload-field${resume ? " has-file" : ""}`}>
          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={(event) => setResume(event.currentTarget.files[0])}
            required
          />
          <span class="upload-icon" aria-hidden="true">↑</span>
          <span class="upload-copy">
            <strong>{resume ? resume.name : "Choose a file"}</strong>
            <small>{resume ? "Ready to continue" : "PDF document only"}</small>
          </span>
        </label>
        {error && <p class="form-error">{error}</p>}
        <div class="button-row">
          <button
            class="text-button"
            type="button"
            onClick={onBack}
            disabled={starting}
          >
            Back
          </button>
          <button
            class="primary-button"
            type="submit"
            disabled={!resume || starting}
          >
            {starting ? "Starting…" : "Start interview"}
            <span aria-hidden="true">→</span>
          </button>
        </div>
      </form>
    </section>
  );
}

function Message({ role, children }) {
  const interviewer = role === "interviewer";

  return (
    <div class={`message-row ${role}`}>
      {interviewer && <img class="avatar" src="logo_btr.png" alt="" />}
      <div class="message">
        <span class="message-author">
          {interviewer ? "Interviewer" : "You"}
        </span>
        <p>{children}</p>
      </div>
    </div>
  );
}

function TypingMessage() {
  return (
    <div class="message-row interviewer">
      <img class="avatar" src="logo_btr.png" alt="" />
      <div class="message typing" aria-label="Interviewer is responding">
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}

function Interview({ email, initialMessage }) {
  const [messages, setMessages] = useState([initialMessage]);
  const [draft, setDraft] = useState("");
  const [responding, setResponding] = useState(false);
  const [complete, setComplete] = useState(false);
  const [error, setError] = useState("");
  const messagesEnd = useRef();

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, responding]);

  async function sendMessage(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || complete) return;

    setMessages((current) => [...current, { role: "applicant", text }]);
    setDraft("");
    setError("");
    setResponding(true);

    try {
      const response = await fetch("api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, content: text }),
      });
      if (!response.ok) throw new Error();

      const result = await response.json();
      setMessages((current) => [
        ...current,
        {
          role: result.interviewer_message.role,
          text: result.interviewer_message.content,
        },
      ]);
      setComplete(result.interview_complete);
    } catch {
      setMessages((current) => current.slice(0, -1));
      setDraft(text);
      setError("The message could not be sent. Please try again.");
    } finally {
      setResponding(false);
    }
  }

  return (
    <section class="interview-shell">
      <header class="interview-header">
        <div>
          <div class={`status-line${complete ? " complete" : ""}`}>
            <span class="status-dot" />
            {complete ? "Interview complete" : "Interview in progress"}
          </div>
          <p>ML Engineer Interview</p>
        </div>
        <Progress step={3} />
      </header>

      <div class="messages" aria-live="polite">
        <div class="conversation-start">Interview started</div>
        {messages.map((message, index) => (
          <Message key={index} role={message.role}>
            {message.text}
          </Message>
        ))}
        {responding && <TypingMessage />}
        <div ref={messagesEnd} />
      </div>

      {error && <p class="chat-error">{error}</p>}
      <form class="composer" onSubmit={sendMessage}>
        <label class="sr-only" for="answer">Your answer</label>
        <textarea
          id="answer"
          rows="1"
          value={draft}
          onInput={(event) => setDraft(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) sendMessage(event);
          }}
          placeholder={complete ? "Responses are closed" : "Write your answer…"}
          disabled={responding || complete}
        />
        <button
          type="submit"
          class="send-button"
          aria-label="Send answer"
          disabled={!draft.trim() || responding || complete}
        >
          ↑
        </button>
      </form>
      <p class="composer-note">
        {complete
          ? "Interview complete · Responses are closed"
          : "Press Enter to send · Shift + Enter for a new line"}
      </p>
    </section>
  );
}

function App() {
  const [step, setStep] = useState(1);
  const [email] = useState(
    () => `${crypto.randomUUID()}@interview.invalid`,
  );
  const [privacyConsent, setPrivacyConsent] = useState(false);
  const [consent, setConsent] = useState(false);
  const [reviewConsent, setReviewConsent] = useState(false);
  const [resume, setResume] = useState();
  const [initialMessage, setInitialMessage] = useState();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  async function startInterview(event) {
    event.preventDefault();
    setStarting(true);
    setError("");

    const form = new FormData();
    form.append("email", email);
    form.append("resume", resume);

    try {
      const response = await fetch("api/applications", {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error();

      const result = await response.json();
      setInitialMessage({
        role: result.interviewer_message.role,
        text: result.interviewer_message.content,
      });
      setStep(3);
    } catch {
      setError("The interview could not be started. Please try again.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <main class={step === 3 ? "app interview-view" : "app"}>
      <header class="brand">
        <img class="brand-mark" src="logo_btr.png" alt="" />
        <span>ML Engineer Interview</span>
      </header>

      {step === 1 && (
        <ConsentStep
          privacyConsent={privacyConsent}
          setPrivacyConsent={setPrivacyConsent}
          consent={consent}
          setConsent={setConsent}
          reviewConsent={reviewConsent}
          setReviewConsent={setReviewConsent}
          onContinue={(event) => {
            event.preventDefault();
            setStep(2);
          }}
        />
      )}

      {step === 2 &&
        (starting ? (
          <ResumeProcessing />
        ) : (
          <ResumeStep
            resume={resume}
            setResume={setResume}
            starting={starting}
            error={error}
            onBack={() => setStep(1)}
            onContinue={startInterview}
          />
        ))}

      {step === 3 && (
        <Interview email={email} initialMessage={initialMessage} />
      )}
    </main>
  );
}

render(<App />, document.querySelector("#app"));
