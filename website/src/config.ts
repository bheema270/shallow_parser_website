import type { OutputFormat, ParsedToken } from "./types/parser";

// Set VITE_API_BASE to override. In development defaults to the FastAPI dev
// server; in production uses same-origin relative URLs (FastAPI serves dist/).
export const API_BASE: string =
  (import.meta.env["VITE_API_BASE"] as string | undefined) ??
  (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

export const OUTPUT_FORMATS: OutputFormat[] = [
  { id: "conllu", label: "CoNLL-U", extension: ".conllu" },
  // Future: { id: "json", label: "JSON", extension: ".json" },
  // Future: { id: "ssf", label: "SSF", extension: ".ssf" },
];

export const TABLE_COLUMNS: { key: keyof ParsedToken; label: string }[] = [
  { key: "id", label: "Token #" },
  { key: "token", label: "Token" },
  { key: "lemma", label: "Lemma" },
  { key: "pos", label: "POS" },
  { key: "lexcat", label: "LexCat" },
  { key: "chunk", label: "Chunk" },
  { key: "person", label: "Person" },
  { key: "number", label: "Number" },
  { key: "gender", label: "Gender" },
  { key: "case", label: "Case" },
  { key: "suffix", label: "Suffix" },
  { key: "tam", label: "TAM" },
];

export const SAMPLE_SENTENCE = "రాముడు బడికి వెళ్ళాడు";
