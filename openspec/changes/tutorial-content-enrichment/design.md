## Context

The tutorial site has 26 chapters. Four chapters (01-market-foundations, 06-trading-strategies, 23-live-trading, 26-realtime-data) already use a multi-page sub-chapter pattern with `index.md` parent pages and numbered child pages (`01-topic.md`, `02-topic.md`). The remaining 22 chapters are single markdown files ranging from 148 to 836 lines. These single-file chapters work but are harder to navigate, lack visual diagrams, and have inconsistent cross-referencing.

The Just the Docs theme supports parent/child navigation via `has_children: true` on parent pages and `parent:` front matter on child pages. Mermaid 10.6.0 is already configured. The project has a dark-fill Mermaid styling convention documented in CLAUDE.md.

## Goals / Non-Goals

**Goals:**
- Every chapter uses multi-page sub-chapter structure (index + 2–4 child pages)
- Every chapter has at least one Mermaid diagram showing architecture, data flow, or decision logic
- Every chapter includes working `from puffin.xxx import ...` code examples
- Every chapter has a "Related Chapters" section linking to 2–4 related chapters
- Consistent chapter template: theory → code → diagram → exercises → summary → related chapters
- One companion Jupyter notebook per chapter cluster (~5 notebooks) with runnable code examples that can be auto-validated via `jupyter nbconvert --execute`

**Non-Goals:**
- Rewriting existing prose that is already good — preserve and redistribute existing content
- Adding new Python modules or changing the `puffin/` package
- Changing the Jekyll theme, site structure, or `_config.yml` (beyond nav_order if needed)
- Perfecting every diagram — simple, accurate diagrams are better than elaborate ones

## Decisions

### 1. Sub-chapter split strategy

**Decision:** Split each single-file chapter into 2–4 sub-pages based on natural topic boundaries, not arbitrary line counts.

**Rationale:** The existing multi-page chapters (01, 06, 23, 26) use 3–6 sub-pages each. Most single-file chapters have clear sections (theory, implementation, advanced usage) that map naturally to 2–4 pages. Forcing uniform page counts would create artificial splits.

**Pattern:**
```
NN-chapter-name/
├── index.md          # has_children: true, overview + chapter listing
├── 01-subtopic.md    # parent: "Part NN: Chapter Name"
├── 02-subtopic.md
└── 03-subtopic.md    # (optional)
```

**Alternative considered:** Keep single files but add internal anchor links — rejected because it doesn't improve navigation and doesn't match the established pattern.

### 2. Handling the existing single-file content

**Decision:** Rename the existing `chapter-name.md` to `index.md`, extract sub-sections into child pages, and update front matter.

**Rationale:** This preserves existing permalinks (the index.md inherits the folder's permalink) and avoids broken links. The existing content is redistributed, not rewritten.

### 3. Mermaid diagram placement

**Decision:** Place the primary diagram in the `index.md` page as an overview/architecture diagram. Add focused diagrams in sub-pages where they clarify a specific flow.

**Rationale:** The index page diagram gives readers a visual map of the chapter's content. Sub-page diagrams go deeper into specific algorithms or pipelines. All diagrams use the project's dark-fill convention (`classDef` with RGB ≤ 0x90, paper background `#d5d0c8`).

### 4. Cross-link structure

**Decision:** Add a `## Related Chapters` section at the bottom of each `index.md` (before Source Code), with 2–4 bullet points linking to related chapters using `{{ site.baseurl }}` URLs.

**Rationale:** Centralizing cross-links on index pages keeps sub-pages focused. Using a consistent section name makes them findable. The link map:

Key cross-link clusters:
- **Data flow**: 02 (data pipeline) → 03 (alt data) → 04 (alpha factors) → 05 (portfolio)
- **ML pipeline**: 08 (linear) → 09 (time series) → 10 (bayesian) → 11 (tree ensembles) → 12 (unsupervised)
- **NLP chain**: 13 (NLP) → 14 (topics) → 15 (embeddings) → 22 (AI-assisted)
- **Deep learning**: 16 (fundamentals) → 17 (CNN) → 18 (RNN) → 19 (autoencoders) → 20 (GANs) → 21 (deep RL)
- **Operational**: 06 (strategies) → 07 (backtesting) → 23 (live trading) → 24 (risk) → 25 (monitoring)

### 5. Processing order

**Decision:** Process chapters in groups by cluster (data → ML → NLP → deep learning → operational), starting with the thinnest chapters (02, 07, 24) as quick wins.

**Rationale:** Working in clusters ensures cross-links within a cluster are consistent. Starting with stubs provides immediate visible progress.

## Risks / Trade-offs

- **Broken internal links** → Preserve existing permalinks by keeping folder-level permalink on `index.md`. Run `bundle exec jekyll build` to verify no broken links after each batch.
- **Large diff size** → Each chapter expansion touches 1 existing file + creates 2–4 new files. Process in batches of 3–4 chapters per commit to keep diffs reviewable.
- **Inconsistent quality across chapters** → Use a checklist per chapter: ☐ index.md with overview diagram ☐ 2–4 sub-pages ☐ working puffin code ☐ exercises ☐ related chapters section.
- **Mermaid rendering issues** → Test diagrams locally with `docker-compose up` before committing. Keep diagrams simple (under 15 nodes).
