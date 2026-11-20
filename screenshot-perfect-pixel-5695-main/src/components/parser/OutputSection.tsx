import { useEffect, useMemo, useState } from "react";
import { Check, Copy, Download } from "lucide-react";
import { OUTPUT_FORMATS, TABLE_COLUMNS } from "@/config";
import type { AnalyzeResponse, ParsedToken } from "@/types/parser";

function FormattedTable({ tokens }: { tokens: ParsedToken[] }) {
  return (
    <div className="overflow-x-auto rounded-sm border border-border">
      <table className="w-full border-collapse font-mono text-sm">
        <thead className="sticky top-0 bg-table-head">
          <tr>
            {TABLE_COLUMNS.map((column) => (
              <th key={column.key} className="whitespace-nowrap border-b border-border px-3 py-2 text-left text-xs font-semibold text-muted-foreground">
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody lang="te">
          {tokens.map((token) => (
            <tr key={`${token.id}-${token.token}`} className="even:bg-stripe odd:bg-card">
              {TABLE_COLUMNS.map((column) => (
                <td key={column.key} className="whitespace-nowrap px-3 py-1.5 font-mono">
                  {String(token[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RawOutput({ raw }: { raw: string }) {
  return (
    <pre lang="te" className="overflow-x-auto rounded-sm border border-border bg-stripe p-4 font-mono text-sm leading-relaxed">
      {raw}
    </pre>
  );
}

export function OutputSection({ data }: { data: AnalyzeResponse }) {
  const [tab, setTab] = useState<"table" | "raw">("table");
  const [selectedSentenceId, setSelectedSentenceId] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const sentences = data.sentences ?? [];
  const selectedSentence = useMemo(
    () => sentences.find((sentence) => sentence.sentence_id === selectedSentenceId),
    [sentences, selectedSentenceId],
  );
  const isWholeDocument = selectedSentence === undefined;
  const displayedTokens = selectedSentence?.tokens ?? data.tokens;
  const displayedRaw = selectedSentence?.raw_output ?? data.raw_output;
  const displayedText = selectedSentence?.sentence ?? data.sentence;
  const ext = OUTPUT_FORMATS.find((format) => format.id === data.format)?.extension ?? ".txt";

  useEffect(() => {
    setSelectedSentenceId(null);
    setTab("table");
    setCopied(false);
  }, [data]);

  const copy = async () => {
    await navigator.clipboard.writeText(displayedRaw);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([displayedRaw], { type: "text/plain;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = isWholeDocument
      ? `complete-output${ext}`
      : `sentence-${selectedSentence?.sentence_id ?? "output"}${ext}`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const tabCls = (active: boolean) =>
    `-mb-px border-b-2 px-1 pb-2 text-sm ${active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`;
  const actCls = "inline-flex h-8 items-center gap-1.5 rounded-sm border border-input px-3 text-xs hover:bg-accent";
  return (
    <section className="space-y-4">
      {sentences.length > 0 && (
        <div className="flex items-center gap-3">
          <label htmlFor="sentence-selector" className="shrink-0 text-xs font-medium text-muted-foreground">View results</label>
          <select
            id="sentence-selector"
            className="h-9 min-w-56 rounded-sm border border-input bg-background px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
            value={selectedSentenceId ?? "all"}
            onChange={(event) => setSelectedSentenceId(event.target.value === "all" ? null : Number(event.target.value))}
          >
            <option value="all">Whole document ({data.tokens.length} tokens)</option>
            {sentences.map((sentence) => {
              const previewTokens = sentence.tokens.slice(0, 2).map((t) => t.token).join(" ");
              const preview = previewTokens ? `"${previewTokens}..." ` : "";
              return (
                <option key={sentence.sentence_id} value={sentence.sentence_id}>
                  Sentence {sentence.sentence_id}: {preview}({sentence.tokens.length} tokens)
                </option>
              );
            })}
          </select>
        </div>
      )}

      <div className="flex items-end justify-between border-b border-border">
        <div className="flex gap-6">
          <button className={tabCls(tab === "table")} onClick={() => setTab("table")}>Formatted View</button>
          <button className={tabCls(tab === "raw")} onClick={() => setTab("raw")}>Raw Output</button>
        </div>
        <div className="mb-2 flex flex-wrap justify-end gap-2">
          <button className={actCls} onClick={copy}>
            {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copied" : isWholeDocument ? "Copy All" : "Copy Sentence"}
          </button>
          <button className={actCls} onClick={download}>
            <Download className="h-3.5 w-3.5" /> {isWholeDocument ? "Download Complete CoNLL-U" : "Download Sentence"}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <p>
          {displayedTokens.length} tokens · <span lang="te">{displayedText}</span>
        </p>
        {data.model_name && (
          <span className="inline-flex items-center rounded-sm bg-muted px-2 py-0.5 text-[11px] font-medium text-foreground">
            Model: {data.model_name}
          </span>
        )}
      </div>
      {tab === "table" ? <FormattedTable tokens={displayedTokens} /> : <RawOutput raw={displayedRaw} />}
    </section>
  );
}
