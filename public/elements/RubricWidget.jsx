import React from "react";

export default function RubricWidget() {
  const rubric = props.judgements;
  console.log("RUBRIC IS!!");
  console.log(rubric);
  const rows = Object.entries(rubric);

  // ---- handle "history" shape while preserving your pattern above ----
  const history = Array.isArray(rubric?.history) ? rubric.history : null;

  const [turn, setTurn] = React.useState(() =>
    history ? Math.max(0, history.length - 1) : 0
  );

  React.useEffect(() => {
    if (!history) return;
    setTurn((prev) => {
      const maxIdx = history.length - 1;
      if (prev >= maxIdx) return maxIdx;
      return Math.max(0, Math.min(prev, maxIdx));
    });
  }, [history?.length]);

  const activeRubric = history ? history[turn] : rubric;
  // -------------------------------------------------------------------

  const clamp01 = (v) => Math.max(0, Math.min(1, Number(v)));

  const normalize3 = (arr) => {
    const a = (arr || []).slice(0, 3);
    const q = a.map(clamp01);
    const s = q.reduce((acc, x) => acc + x, 0);
    if (!s) return [1 / 3, 1 / 3, 1 / 3];
    return q.map((x) => x / s);
  };

  const normalizedEntropy = (ps) => {
    const K = ps.length;
    const logK = Math.log(K);
    const H = -ps.reduce((acc, p) => (p > 0 ? acc + p * Math.log(p) : acc), 0);
    return H / logK;
  };

  const BAR_COLORS = ["hsl(0 85% 45%)", "hsl(35 90% 45%)", "hsl(120 70% 35%)"];
  const LEVEL_LABELS = ["Low", "Med", "High"];

  // NEW: read from new schema
  const posteriors = activeRubric?.posteriors || {};
  const justifications = activeRubric?.justifications || {};

  // We iterate criteria from posteriors keys (instead of Object.entries(activeRubric))
  const criteriaRows = Object.entries(posteriors);

  const prettyLabel = (s) =>
    String(s || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

  // ---- custom tooltip (NO react-dom / no new modules) ----
  const DEBUG_TOOLTIP = true;

  const TOOLTIP_MAX_W = 320;
  const TOOLTIP_EST_H = 120;

  const [tooltip, setTooltip] = React.useState({
    open: false,
    x: 0,
    y: 0,
    content: "",
  });

  // Create ONE tooltip DOM node attached to document.body
  const tooltipElRef = React.useRef(null);

  React.useEffect(() => {
    if (typeof document === "undefined") return;

    const el = document.createElement("div");
    el.setAttribute("role", "tooltip");

    // base styles (kept here so it doesn't depend on React render)
    Object.assign(el.style, {
      position: "fixed",
      zIndex: "2147483647",
      pointerEvents: "none",
      maxWidth: `${TOOLTIP_MAX_W}px`,
      padding: "8px 10px",
      borderRadius: "10px",
      background: "rgba(20,20,20,0.92)",
      color: "rgba(255,255,255,0.96)",
      fontSize: "14px",
      lineHeight: "1.35",
      boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
      whiteSpace: "pre-wrap",
      display: "none",
      left: "0px",
      top: "0px",
    });

    document.body.appendChild(el);
    tooltipElRef.current = el;

    return () => {
      try {
        el.remove();
      } catch {
        // ignore
      }
      tooltipElRef.current = null;
    };
  }, []);

  const clampPos = React.useCallback((x, y) => {
    const vw = typeof window !== "undefined" ? window.innerWidth : 0;
    const vh = typeof window !== "undefined" ? window.innerHeight : 0;

    const left = vw
      ? Math.max(8, Math.min(x + 12, vw - 8 - TOOLTIP_MAX_W))
      : x + 12;

    const top = vh
      ? Math.max(8, Math.min(y + 12, vh - 8 - TOOLTIP_EST_H))
      : y + 12;

    return { left, top };
  }, []);

  // Keep tooltip DOM node synced with state
  React.useEffect(() => {
    const el = tooltipElRef.current;
    if (!el) return;

    if (tooltip.open && tooltip.content) {
      const { left, top } = clampPos(tooltip.x, tooltip.y);
      el.textContent = tooltip.content;
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      el.style.display = "block";
    } else {
      el.style.display = "none";
    }
  }, [tooltip.open, tooltip.content, tooltip.x, tooltip.y, clampPos]);

  // IMPORTANT: open/update on BOTH mouseenter and mousemove
  const showTooltip = React.useCallback(
    (content, e) => {
      const text = String(content || "");
      if (DEBUG_TOOLTIP) {
        console.log("[RubricWidget tooltip] showTooltip fired", {
          content: text,
          type: e?.type,
          clientX: e?.clientX,
          clientY: e?.clientY,
          target: e?.target,
        });
      }

      if (!text) return;

      const x = e?.clientX ?? 0;
      const y = e?.clientY ?? 0;

      setTooltip((t) => {
        if (t.open && t.content === text) return { ...t, x, y };
        return { open: true, x, y, content: text };
      });
    },
    [DEBUG_TOOLTIP]
  );

  const hideTooltip = React.useCallback(
    (e) => {
      if (DEBUG_TOOLTIP) {
        console.log("[RubricWidget tooltip] hideTooltip fired", {
          type: e?.type,
          target: e?.target,
        });
      }
      setTooltip((t) => (t.open ? { ...t, open: false, content: "" } : t));
    },
    [DEBUG_TOOLTIP]
  );
  // --------------------------------------------------------

  return (
    <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
      {/* turn selector */}
      {history && history.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          <div style={{ fontSize: 12, fontWeight: 700, opacity: 0.75, marginRight: 6 }}>
            Turn
          </div>
          {history.map((_, i) => {
            const active = i === turn;
            return (
              <button
                key={i}
                onClick={() => setTurn(i)}
                onMouseEnter={(e) => showTooltip(`View rubric at turn ${i}`, e)}
                onMouseMove={(e) => showTooltip(`View rubric at turn ${i}`, e)}
                onMouseLeave={hideTooltip}
                style={{
                  border: "none",
                  cursor: "pointer",
                  padding: "4px 8px",
                  borderRadius: 999,
                  fontSize: 12,
                  fontWeight: 700,
                  background: active ? "rgba(0,0,0,0.12)" : "rgba(0,0,0,0.06)",
                }}
              >
                {i}
              </button>
            );
          })}
        </div>
      )}

      {/* NEW: render from posteriors/justifications */}
      {criteriaRows.map(([key, dist]) => {
        const ps = normalize3([dist?.low, dist?.medium, dist?.high]);
        const e01 = normalizedEntropy(ps);
        const entPct = Math.round(e01 * 100);

        const justification = justifications?.[key] || "";

        return (
          <div
            key={key}
            style={{ display: "flex", flexDirection: "column", gap: 8 }}
            onMouseEnter={(e) => {
              if (DEBUG_TOOLTIP) console.log("[RubricWidget] row onMouseEnter", key, e?.type);
              showTooltip(justification, e);
            }}
            onMouseMove={(e) => {
              if (DEBUG_TOOLTIP) console.log("[RubricWidget] row onMouseMove", key, e?.type);
              showTooltip(justification, e); // opens even if mouseenter doesn't fire
            }}
            onMouseLeave={(e) => {
              if (DEBUG_TOOLTIP) console.log("[RubricWidget] row onMouseLeave", key, e?.type);
              hideTooltip(e);
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
              <div
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {prettyLabel(key)}
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gap: 10 }}>
              {ps.map((p, i) => {
                const height = Math.round(p * 44);
                const pct = Math.round(p * 100);
                return (
                  <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
                    <div
                      style={{
                        height: 44,
                        borderRadius: 10,
                        background: "rgba(0,0,0,0.08)",
                        overflow: "hidden",
                        display: "flex",
                        alignItems: "flex-end",
                      }}
                    >
                      <div
                        style={{
                          width: "100%",
                          height,
                          background: BAR_COLORS[i],
                          transition: "height 200ms ease",
                        }}
                      />
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", gap: 6 }}>
                      <div style={{ fontSize: 11, opacity: 0.75 }}>{LEVEL_LABELS[i]}</div>
                      <div style={{ fontSize: 11, fontVariantNumeric: "tabular-nums", opacity: 0.85 }}>
                        {pct}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}