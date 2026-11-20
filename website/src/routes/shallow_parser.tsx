import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Footer, Header } from "@/components/parser/Header";
import { InputSection } from "@/components/parser/InputSection";
import { OutputSection } from "@/components/parser/OutputSection";
import { OUTPUT_FORMATS } from "@/config";
import { useAnalyze } from "@/hooks/useAnalyze";
import { useModels } from "@/hooks/useModels";

export const Route = createFileRoute("/shallow_parser")({
  head: () => ({
    meta: [
      { title: "Telugu Shallow Parser — LTRC, IIIT Hyderabad" },
      { name: "description", content: "Morphological and syntactic annotation of Telugu sentences: POS, chunk, lemma, case, TAM in CoNLL-U." },
      { property: "og:title", content: "Telugu Shallow Parser" },
      { property: "og:description", content: "Analyze Telugu sentences and export CoNLL-U annotations." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ShallowParserPage,
});

export function ShallowParserPage() {
  const [text, setText] = useState("");
  const [format, setFormat] = useState(OUTPUT_FORMATS[0]?.id ?? "conllu");
  const { models, selectedModel, setSelectedModel } = useModels();
  const { data, loading, error, analyze, cancel, reset } = useAnalyze();

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="mx-auto w-full max-w-[900px] flex-1 space-y-12 px-6 py-12">
        <InputSection
          text={text}
          setText={setText}
          format={format}
          setFormat={setFormat}
          models={models}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          loading={loading}
          error={error}
          onAnalyze={() => analyze(text.trim(), format, selectedModel)}
          onCancel={cancel}
          onClear={() => { setText(""); reset(); }}
        />
        {data && <OutputSection data={data} />}
      </main>
      <Footer />
    </div>
  );
}
