from abc import ABC, abstractmethod


class OutputFormatter(ABC):
    label: str = ""
    extension: str = ""

    @abstractmethod
    def format(self, tokens: list[dict], sentence: str) -> str: ...


class CoNLLUFormatter(OutputFormatter):
    label = "CoNLL-U"
    extension = ".conllu"

    def format(self, tokens: list[dict], sentence: str) -> str:
        def _clean(val) -> str:
            s = str(val) if val is not None else "_"
            if s in {"-", "UNK", "[UNK]", "<UNK>", "None", ""}:
                return "_"
            return s.replace("<UNK>", "_").replace("[UNK]", "_")

        lines = [f"# text = {sentence}"]
        for tok in tokens:
            fields = [
                str(tok["id"]),
                str(tok["token"]),
                str(tok["lemma"]),
                str(tok["pos"]),
                str(tok["lexcat"]),
                str(tok["chunk"]),
                _clean(tok.get("person")),
                _clean(tok.get("number")),
                _clean(tok.get("gender")),
                _clean(tok.get("case")),
                _clean(tok.get("suffix")),
                _clean(tok.get("tam")),
            ]
            lines.append("\t".join(fields))
        return "\n".join(lines)

    def format_document(self, sentences: list[dict]) -> str:
        blocks = []
        for s in sentences:
            blocks.append(self.format(s["tokens"], s["sentence"]))
        return "\n\n".join(blocks)


# Registry — easy to extend
FORMATTERS: dict[str, OutputFormatter] = {
    "conllu": CoNLLUFormatter(),
}
