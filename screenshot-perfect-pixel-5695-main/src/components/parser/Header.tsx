export function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-background">
      <div className="mx-auto flex h-14 max-w-[900px] items-center justify-between px-6">
        <h1 className="text-base font-medium tracking-tight">Telugu Shallow Parser</h1>
        <a
          href="https://ltrc.iiit.ac.in/"
          target="_blank"
          rel="noreferrer"
          className="text-sm text-muted-foreground hover:text-primary"
        >
          IIIT Hyderabad
        </a>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto max-w-[900px] px-6 py-6 text-xs text-muted-foreground">
        Language Technologies Research Centre, IIIT Hyderabad
      </div>
    </footer>
  );
}
