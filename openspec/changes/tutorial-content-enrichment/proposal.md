## Why

The tutorial site has 26 chapters but 22 of them are single-file pages (148–836 lines each), while only 4 chapters (01, 06, 23, 26) have been expanded into multi-page sub-chapters with index navigation. The single-file chapters lack Mermaid architecture diagrams, have minimal cross-linking to related chapters, and don't consistently show working `puffin` module code examples. Enriching these chapters will make the guide more navigable, visually informative, and practical.

## What Changes

- **Expand thin chapters into multi-page sub-chapters**: Break the 22 single-file chapters into 2–4 focused sub-pages each, with an `index.md` parent page, mirroring the pattern in `01-market-foundations/` and `23-live-trading/`
- **Add Mermaid diagrams**: Add architecture, pipeline, and decision-flow diagrams to each chapter (at least 1 per chapter) using the project's dark-fill Mermaid styling convention
- **Add inline code examples**: Ensure every chapter has working `puffin` module import examples showing real API usage, not just conceptual pseudocode
- **Cross-link related chapters**: Add "Related Chapters" / "See Also" sections and inline links connecting related content (e.g., alpha factors → tree ensembles that consume them, backtesting → strategies that feed into it, NLP sentiment → AI-assisted trading)
- **Standardize chapter structure**: Every chapter follows the template: theory → code → diagram → exercises → summary → related chapters
- **Add cluster notebooks**: One companion Jupyter notebook per chapter cluster (~5 notebooks: data, ML, NLP, deep learning, operational) with runnable code examples, auto-validatable via `jupyter nbconvert --execute`

## Capabilities

### New Capabilities
- `tutorial-chapter-expansion`: Multi-page sub-chapter structure for all 22 single-file tutorial chapters, with index pages and Just the Docs parent/child navigation
- `tutorial-diagrams`: Mermaid architecture and pipeline diagrams across all tutorial chapters, following the project's dark-fill styling convention
- `tutorial-cross-links`: Systematic cross-referencing between related tutorial chapters via "Related Chapters" sections and inline links
- `tutorial-code-examples`: Working inline Python code examples using actual `puffin` module APIs in every tutorial chapter
- `tutorial-cluster-notebooks`: One Jupyter notebook per chapter cluster (~5 total) with runnable, auto-validatable code examples

### Modified Capabilities
<!-- No existing specs are being modified — this is purely tutorial content work -->

## Impact

- **docs/**: All 22 single-file chapter folders gain sub-chapter `.md` files and `index.md` pages
- **docs/_config.yml**: May need nav_order adjustments for new child pages
- **No Python code changes**: This is a docs-only change — no modifications to `puffin/` or `tests/`
- **Build**: More pages means longer Jekyll build time but no dependency changes
