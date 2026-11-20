export function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-background/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-[900px] items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <a
            href="https://ltrc.iiit.ac.in/"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 hover:opacity-85 transition-opacity"
            title="IIIT Hyderabad - LTRC"
          >
            <img
              src="/iiit-logo.jpg"
              alt="IIIT Hyderabad Logo"
              className="h-9 w-auto object-contain rounded-sm"
            />
          </a>
          <div className="h-6 w-px bg-border mx-1" />
          <h1 className="text-base font-semibold tracking-tight text-foreground">
            Telugu Shallow Parser
          </h1>
        </div>
        <a
          href="https://ltrc.iiit.ac.in/"
          target="_blank"
          rel="noreferrer"
          className="text-xs font-medium text-muted-foreground hover:text-primary transition-colors flex items-center gap-1.5"
        >
          <span>LTRC, IIIT Hyderabad</span>
        </a>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-border bg-card/50">
      <div className="mx-auto flex max-w-[900px] items-center justify-between px-6 py-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <img
            src="/iiit-logo.jpg"
            alt="IIIT Hyderabad"
            className="h-6 w-auto object-contain opacity-75"
          />
          <span>Language Technologies Research Centre, IIIT Hyderabad</span>
        </div>
        <span>Telugu Shallow Parser Platform</span>
      </div>
    </footer>
  );
}
