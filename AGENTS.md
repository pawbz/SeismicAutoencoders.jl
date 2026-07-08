## Pluto notebooks (live collaborative sessions)

Notebooks (`*.jl` with Pluto cell markers) may be OPEN in a live lazy-mode Pluto server.
- Edit notebook files directly with your file tools. Edits only mark cells stale — nothing
  runs until requested, and the human sees staleness live in their browser.
- `pluto-collab status <nb.jl>` shows per-cell state and outputs.
- `pluto-collab run <nb.jl> --stale` runs exactly what's outdated (blocking; exit 1 if a
  cell errors). Never re-run the whole notebook.
- `pluto-collab restart <nb.jl>` restarts the kernel and re-runs everything — use it ONLY to
  recover a dead/exited worker ("Process exited" / `TerminatedWorkerException`), which interrupt
  and run cannot revive.
- All cell outputs are also in `<nb.jl>.pluto-cache.toml` (plain TOML; a deletable cache).
- Cell ids are the UUIDs in `# ╔═╡ <uuid>` markers. Keep the `# ╔═╡ Cell order:` section
  in sync when adding/removing cells.
