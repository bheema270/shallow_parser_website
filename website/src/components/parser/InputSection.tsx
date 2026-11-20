import { OUTPUT_FORMATS, SAMPLE_SENTENCE } from "@/config";
import type { ModelOption } from "@/types/parser";

interface Props {
  text: string;
  setText: (v: string) => void;
  format: string;
  setFormat: (v: string) => void;
  models?: ModelOption[];
  selectedModel?: string;
  setSelectedModel?: (v: string) => void;
  loading: boolean;
  error: string | null;
  onAnalyze: () => void;
  onCancel?: () => void;
  onClear: () => void;
}

export function InputSection({
  text,
  setText,
  format,
  setFormat,
  models,
  selectedModel,
  setSelectedModel,
  loading,
  error,
  onAnalyze,
  onCancel,
  onClear,
}: Props) {
  const empty = text.trim().length === 0;
  return (
    <section className="space-y-3">
      <label htmlFor="input" className="block text-sm font-medium">
        Input sentence
      </label>
      <textarea
        id="input"
        lang="te"
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !empty && !loading) onAnalyze();
        }}
        placeholder="తెలుగు వాక్యాన్ని ఇక్కడ టైపు చేయండి..."
        className="w-full resize-y rounded-sm border border-input bg-card px-3 py-2 text-base leading-relaxed outline-none focus:border-ring"
      />
      <div className="flex flex-wrap items-center gap-3">
        {models && models.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            Model
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel?.(e.target.value)}
              disabled={loading}
              className="h-9 rounded-sm border border-input bg-card px-2 text-sm text-foreground outline-none focus:border-ring"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          Format
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            disabled={loading}
            className="h-9 rounded-sm border border-input bg-card px-2 text-sm text-foreground outline-none focus:border-ring"
          >
            {OUTPUT_FORMATS.map((f) => (
              <option key={f.id} value={f.id}>{f.label}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setText(SAMPLE_SENTENCE)}
          disabled={loading}
          className="text-sm text-muted-foreground underline-offset-4 hover:text-primary hover:underline disabled:opacity-50"
        >
          Try example
        </button>
        <div className="ml-auto flex items-center gap-2">
          {loading && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="h-9 rounded-sm border border-destructive/50 text-destructive px-3 text-sm hover:bg-destructive/10"
              title="Halt active analysis"
            >
              Halt / Cancel
            </button>
          )}
          <button
            type="button"
            onClick={onClear}
            disabled={loading}
            className="h-9 rounded-sm border border-input px-4 text-sm hover:bg-accent disabled:opacity-50"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={onAnalyze}
            disabled={empty || loading}
            className="inline-flex h-9 items-center gap-2 rounded-sm bg-primary px-5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading && (
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" />
            )}
            {loading ? "Analyzing..." : <>Analyze <span lang="te" className="text-xs opacity-75">విశ్లేషించు</span></>}
          </button>
        </div>
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </section>
  );
}
