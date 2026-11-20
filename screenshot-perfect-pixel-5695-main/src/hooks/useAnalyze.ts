import { useCallback, useRef, useState } from "react";
import { API_BASE } from "@/config";
import type { AnalyzeResponse } from "@/types/parser";

export const ERR_ANALYSIS = "Analysis failed. Please check your input or try again.";
export const ERR_NETWORK = "Unable to reach the server. Please ensure the backend is running.";

export function useAnalyze() {
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const isCancelledRef = useRef<boolean>(false);

  const cancel = useCallback(async () => {
    isCancelledRef.current = true;
    const jobId = activeJobIdRef.current;
    if (jobId) {
      try {
        await fetch(`${API_BASE}/api/analyze/${encodeURIComponent(jobId)}/cancel`, {
          method: "POST",
        });
      } catch (err) {
        console.warn("Could not notify server of cancellation", err);
      }
    }
    setLoading(false);
    setError("Analysis was halted by user.");
  }, []);

  const analyze = useCallback(async (text: string, format: string, modelId?: string) => {
    setLoading(true);
    setError(null);
    isCancelledRef.current = false;
    activeJobIdRef.current = null;
    const url = `${API_BASE}/api/analyze`;
    console.info("Analyze request starting", { apiBase: API_BASE, url, characters: text.length, format, modelId });
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, format, model_id: modelId }),
      });
      console.info("Analyze queue response received", {
        url,
        status: res.status,
        contentType: res.headers.get("content-type"),
        allowOrigin: res.headers.get("access-control-allow-origin"),
      });
    } catch (err) {
      console.error("Analyze request failed", { apiBase: API_BASE, url, error: err });
      setError(ERR_NETWORK);
      setLoading(false);
      return;
    }
    try {
      const json = await res.json();
      if (!res.ok) {
        console.error("Analyze request returned an error response", { url, status: res.status, detail: json?.detail });
        setError(json?.detail || ERR_ANALYSIS);
        return;
      }
      if (typeof json?.job_id !== "string") {
        console.error("Analyze API did not return a job ID", { url });
        setError("The server returned an invalid analysis job response.");
        return;
      }

      const jobId = json.job_id as string;
      activeJobIdRef.current = jobId;
      console.info("Analysis queued", { jobId, status: json.status });
      const pollUrl = `${API_BASE}/api/analyze/${encodeURIComponent(jobId)}`;
      const deadline = Date.now() + 15 * 60 * 1000;

      while (Date.now() < deadline) {
        if (isCancelledRef.current) {
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        if (isCancelledRef.current) {
          return;
        }
        const statusResponse = await fetch(pollUrl, { method: "GET", cache: "no-store" });
        if (!statusResponse.ok) {
          const statusError = await statusResponse.json().catch(() => ({}));
          throw new Error(statusError?.detail || `Status request failed (${statusResponse.status})`);
        }
        const status = await statusResponse.json();
        if (status.status === "cancelled") {
          setError("Analysis was cancelled.");
          return;
        }
        if (status.status === "failed") throw new Error(status.detail || "Analysis failed.");
        if (status.status === "complete") {
          const result = status.result as AnalyzeResponse;
          console.info("Analyze response parsed", {
            jobId,
            tokens: Array.isArray(result?.tokens) ? result.tokens.length : "missing",
            sentences: Array.isArray(result?.sentences) ? result.sentences.length : "missing",
          });
          setData(result);
          return;
        }
      }
      throw new Error("Analysis is taking longer than 15 minutes. Please try again.");
    } catch (err) {
      if (!isCancelledRef.current) {
        console.error("Analysis polling failed", { url, error: err });
        setError(err instanceof Error ? err.message : ERR_ANALYSIS);
      }
    } finally {
      activeJobIdRef.current = null;
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, loading, error, analyze, cancel, reset };
}
