# User Preferences & Workflow Guidelines

## Tooling

- **Package Manager:** `uv` (avoid `pip` / `venv` unless necessary).
- **Task Runner:** `just` (use `justfile`, never `Makefile`).
- **Ticketing:** `bd` (Epics > Features > Tasks).

## Coding Style

- **Paradigm:** Functional > OOP. Use classes only when frameworks (e.g., DSPy) strictly enforce it.
- **Async/IO:** Always use `async/await` and `httpx`. Avoid `requests`.
- **Resilience:** Implement explicit retries, fallbacks, and robust error handling for external services.
- **Colocation:** "What changes together belongs together." Group code by feature/domain, not by technical layer. Avoid scattering related logic across multiple "utils" or "helpers" files.
- **WTFs/Minute:** Minimize "WTFs/Minute". Write clean, standard, unsurprising code. Refactor hacks immediately.

## Workflow

- **Planning:** Brainstorm architecture and "happy paths" before coding. Use `bd` for all task tracking. Do NOT use internal todo tools/memory.
- **Verification:** Always test immediately (e.g., `curl`, `pytest`, `just test`) after implementation.
- **Communication:** Be concise. Focus on "Changes Made" and "Next Steps". Avoid verbose pleasantries.
- **Boy Scout Rule:** If you spot issues (tech debt, messiness, bugs) while working, create a ticket immediately. Capture improvements as you go.
