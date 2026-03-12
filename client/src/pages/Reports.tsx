/**
 * OLEA Intelligence — Reports Page
 *
 * Issue C fixes:
 *   - No manual refetch() on preview toggle.
 *     The query re-enables naturally via `enabled: showPreview`.
 *   - Only the explicit "Ανανέωση" (Refresh) button calls refetch().
 *
 * Issue G fixes:
 *   - All routes use singular /api/report/... (not /api/reports/...)
 *   - /api/report/pdf, /api/report/weekly-pdf, /api/report/send, /api/report/preview
 *
 * Dependencies (client-side, not in server tsconfig):
 *   react, react-dom, @tanstack/react-query
 */

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

// ─── Types ────────────────────────────────────────────────────────────────────
interface SendResult {
  ok: boolean;
  sentAt: string;
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function fetchPreviewHtml(): Promise<string> {
  const res = await fetch("/api/report/preview");
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Preview failed (${res.status}): ${body}`);
  }
  return res.text();
}

async function sendReport(): Promise<SendResult> {
  const res = await fetch("/api/report/send", { method: "POST" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Send failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<SendResult>;
}

// ─── Component ────────────────────────────────────────────────────────────────
export function OleaEmailReporter(): React.ReactElement {
  const [showPreview, setShowPreview] = useState(false);
  const [sendState, setSendState] = useState<
    "idle" | "sending" | "sent" | "error"
  >("idle");
  const [sendError, setSendError] = useState<string | null>(null);

  // Issue C: no manual refetch() on toggle.
  // `enabled: showPreview` drives the fetch automatically when toggled on.
  // Only the Refresh button below calls refetch() explicitly.
  const {
    data: previewHtml,
    refetch,
    isFetching,
    isError,
    error,
  } = useQuery({
    queryKey: ["olea-report-preview"],
    queryFn: fetchPreviewHtml,
    enabled: showPreview,
    staleTime: 0, // always re-fetch fresh when re-enabled
    gcTime: 5 * 60 * 1000,
    retry: 1,
  });

  // Toggle preview — let enabled: showPreview drive the query (Issue C)
  function handleTogglePreview(): void {
    setShowPreview((prev) => !prev);
    // Do NOT call refetch() here — the query handles it via `enabled`
  }

  // Explicit refresh — only when preview is already visible
  function handleRefresh(): void {
    void refetch();
  }

  async function handleSend(): Promise<void> {
    setSendState("sending");
    setSendError(null);
    try {
      await sendReport();
      setSendState("sent");
    } catch (err) {
      setSendError(err instanceof Error ? err.message : String(err));
      setSendState("error");
    }
  }

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>OLEA Intelligence</h1>
        <p style={styles.subtitle}>Εβδομαδιαία Αναφορά Αγοράς Ελαιόλαδου</p>
      </header>

      {/* ── Action bar ──────────────────────────────────────────────── */}
      <div style={styles.actionBar}>
        {/* Preview toggle */}
        <button style={styles.btnPrimary} onClick={handleTogglePreview}>
          {showPreview ? "Απόκρυψη Προεπισκόπησης" : "Εμφάνιση Προεπισκόπησης"}
        </button>

        {/* Refresh — only calls refetch(); does not toggle showPreview */}
        {showPreview && (
          <button
            style={styles.btnSecondary}
            onClick={handleRefresh}
            disabled={isFetching}
          >
            {isFetching ? "Φόρτωση…" : "Ανανέωση"}
          </button>
        )}

        {/* Issue G: /api/report/pdf (singular) */}
        <a
          href="/api/report/pdf"
          download={`olea-report-${today()}.pdf`}
          style={styles.btnDownload}
        >
          Λήψη PDF
        </a>

        {/* Issue G: /api/report/weekly-pdf (singular, new route) */}
        <a
          href="/api/report/weekly-pdf"
          download={`olea-weekly-report-${today()}.pdf`}
          style={styles.btnDownload}
        >
          Λήψη Εβδομαδιαίας Αναφοράς (PDF)
        </a>

        {/* Send report */}
        <button
          style={
            sendState === "sending" ? styles.btnDisabled : styles.btnSend
          }
          onClick={() => void handleSend()}
          disabled={sendState === "sending"}
        >
          {sendState === "sending" ? "Αποστολή…" : "Αποστολή Αναφοράς"}
        </button>
      </div>

      {/* ── Send feedback ────────────────────────────────────────────── */}
      {sendState === "sent" && (
        <div style={styles.alertSuccess}>
          ✓ Η αναφορά απεστάλη επιτυχώς.
        </div>
      )}
      {sendState === "error" && (
        <div style={styles.alertError}>
          ✗ Σφάλμα αποστολής: {sendError}
        </div>
      )}

      {/* ── Preview pane ─────────────────────────────────────────────── */}
      {showPreview && (
        <div style={styles.previewContainer}>
          {isFetching && !previewHtml && (
            <div style={styles.statusMsg}>Φόρτωση προεπισκόπησης…</div>
          )}
          {isError && (
            <div style={styles.alertError}>
              ✗ Σφάλμα κατά τη φόρτωση:{" "}
              {error instanceof Error ? error.message : "Άγνωστο σφάλμα"}
            </div>
          )}
          {previewHtml && (
            <iframe
              srcDoc={previewHtml}
              title="Προεπισκόπηση Αναφοράς OLEA"
              style={styles.previewFrame}
              sandbox="allow-same-origin"
            />
          )}
        </div>
      )}
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function today(): string {
  return new Date().toISOString().slice(0, 10);
}

// ─── Inline styles ────────────────────────────────────────────────────────────
const styles = {
  page: {
    fontFamily: "Arial, Helvetica, sans-serif",
    maxWidth: 1100,
    margin: "0 auto",
    padding: "24px 20px",
    color: "#222",
  } as React.CSSProperties,

  header: {
    marginBottom: 24,
    borderBottom: "2px solid #2c5f2e",
    paddingBottom: 12,
  } as React.CSSProperties,

  title: {
    fontSize: 22,
    color: "#2c5f2e",
    marginBottom: 4,
  } as React.CSSProperties,

  subtitle: {
    fontSize: 13,
    color: "#666",
  } as React.CSSProperties,

  actionBar: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: 10,
    marginBottom: 20,
  } as React.CSSProperties,

  btnPrimary: {
    background: "#2c5f2e",
    color: "#fff",
    border: "none",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
  } as React.CSSProperties,

  btnSecondary: {
    background: "#fff",
    color: "#2c5f2e",
    border: "1px solid #2c5f2e",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
  } as React.CSSProperties,

  btnDownload: {
    background: "#e8f5e9",
    color: "#2c5f2e",
    border: "1px solid #a5d6a7",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
    textDecoration: "none",
    display: "inline-flex",
    alignItems: "center",
  } as React.CSSProperties,

  btnSend: {
    background: "#1565c0",
    color: "#fff",
    border: "none",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
  } as React.CSSProperties,

  btnDisabled: {
    background: "#90a4ae",
    color: "#fff",
    border: "none",
    padding: "8px 16px",
    borderRadius: 4,
    cursor: "not-allowed",
    fontSize: 13,
    fontWeight: 600,
  } as React.CSSProperties,

  alertSuccess: {
    background: "#e8f5e9",
    border: "1px solid #a5d6a7",
    color: "#1b5e20",
    padding: "10px 16px",
    borderRadius: 4,
    marginBottom: 16,
    fontSize: 13,
  } as React.CSSProperties,

  alertError: {
    background: "#ffebee",
    border: "1px solid #ef9a9a",
    color: "#b71c1c",
    padding: "10px 16px",
    borderRadius: 4,
    marginBottom: 16,
    fontSize: 13,
  } as React.CSSProperties,

  previewContainer: {
    marginTop: 8,
    border: "1px solid #d4e6d4",
    borderRadius: 4,
    overflow: "hidden",
  } as React.CSSProperties,

  previewFrame: {
    width: "100%",
    height: 800,
    border: "none",
    display: "block",
  } as React.CSSProperties,

  statusMsg: {
    padding: 20,
    textAlign: "center" as const,
    color: "#666",
    fontSize: 13,
  } as React.CSSProperties,
} as const;
