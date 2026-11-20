export interface ParsedToken {
  id: number;
  token: string;
  lemma: string;
  pos: string;
  lexcat: string;
  chunk: string;
  person: string;
  number: string;
  gender: string;
  case: string;
  suffix: string;
  tam: string;
}

export interface ModelOption {
  id: string;
  label: string;
  is_default: boolean;
  model_file: string;
}

export interface AnalyzeRequest {
  text: string;
  format: string;
  model_id?: string;
}

export interface SentenceResult {
  sentence_id: number;
  sentence: string;
  tokens: ParsedToken[];
  raw_output: string;
}

export interface AnalyzeResponse {
  tokens: ParsedToken[];
  raw_output: string;
  format: string;
  sentence: string;
  sentences?: SentenceResult[];
  model_used?: string;
  model_name?: string;
}

export interface OutputFormat {
  id: string;
  label: string;
  extension: string;
}
