/**
 * OLEA Intelligence — Report HTTP route handler
 *
 * All report routes use the singular /api/report/... prefix (Issue G).
 *
 * Routes:
 *   GET  /api/report/preview     → generatePreviewReport() — read-only, no side effects
 *   GET  /api/report/pdf         → preview HTML → real Playwright PDF (Issue A)
 *   GET  /api/report/weekly-pdf  → publish HTML → real Playwright PDF (Issue G)
 *   POST /api/report/send        → generatePublishReport() — full pipeline
 *
 * Integration:
 *   Call handleReportRequest(url, req, res) from your http.createServer handler
 *   before your own routing logic. It returns true if the request was handled.
 *
 *   Example (server/app.ts):
 *     import { handleReportRequest } from "./routes";
 *     const server = http.createServer(async (req, res) => {
 *       if (req.url && await handleReportRequest(new URL(req.url, base), req, res)) return;
 *       // ... existing routes
 *     });
 */

import type { IncomingMessage, ServerResponse } from "http";
import type { URL } from "url";

import {
  generatePreviewReport,
  generatePublishReport,
  generatePdfReport,
} from "./analytics/email-reporter";

function isoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function sendError(
  res: ServerResponse,
  status: number,
  message: string
): void {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify({ error: message }));
}

/**
 * Handles all /api/report/* routes.
 * @returns true if the request was handled, false if no route matched.
 */
export async function handleReportRequest(
  url: URL,
  req: IncomingMessage,
  res: ServerResponse
): Promise<boolean> {
  const path = url.pathname;
  const method = req.method ?? "GET";

  // ── GET /api/report/preview ──────────────────────────────────────────────────
  // Issue B: uses generatePreviewReport() — read-only, no side effects
  if (path === "/api/report/preview" && method === "GET") {
    try {
      const html = await generatePreviewReport();
      res.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
      });
      res.end(html);
    } catch (err) {
      console.error("[routes] /api/report/preview error:", err);
      sendError(res, 500, "Preview generation failed");
    }
    return true;
  }

  // ── GET /api/report/pdf ──────────────────────────────────────────────────────
  // Issue A: real PDF via Playwright; Issue B: uses preview (read-only) path
  if (path === "/api/report/pdf" && method === "GET") {
    try {
      const html = await generatePreviewReport();
      const pdfBuffer = await generatePdfReport(html);
      res.writeHead(200, {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="olea-report-${isoDate()}.pdf"`,
        "Cache-Control": "no-store",
      });
      res.end(pdfBuffer);
    } catch (err) {
      console.error("[routes] /api/report/pdf error:", err);
      sendError(res, 500, "PDF generation failed");
    }
    return true;
  }

  // ── GET /api/report/weekly-pdf ───────────────────────────────────────────────
  // Issue G: route added; uses publish path (full pipeline) → real PDF
  if (path === "/api/report/weekly-pdf" && method === "GET") {
    try {
      const { html } = await generatePublishReport();
      const pdfBuffer = await generatePdfReport(html);
      res.writeHead(200, {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="olea-weekly-report-${isoDate()}.pdf"`,
        "Cache-Control": "no-store",
      });
      res.end(pdfBuffer);
    } catch (err) {
      console.error("[routes] /api/report/weekly-pdf error:", err);
      sendError(res, 500, "Weekly PDF generation failed");
    }
    return true;
  }

  // ── POST /api/report/send ────────────────────────────────────────────────────
  // Issue B: uses generatePublishReport() — full pipeline with side effects
  if (path === "/api/report/send" && method === "POST") {
    try {
      const result = await generatePublishReport();
      res.writeHead(200, {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      });
      res.end(JSON.stringify({ ok: true, sentAt: result.sentAt }));
    } catch (err) {
      console.error("[routes] /api/report/send error:", err);
      sendError(res, 500, "Report send failed");
    }
    return true;
  }

  return false;
}
