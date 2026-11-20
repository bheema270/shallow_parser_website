<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Project rules
- Frontend is the TanStack app; the Python FastAPI backend lives in `backend/` and is run separately by the user — the frontend calls it via `API_BASE` in `src/config.ts` (keeps model inference out of the Worker runtime).
- Output formats and table columns are config arrays in `src/config.ts`; add formats there and in `backend/formatter.py` FORMATTERS registry.
