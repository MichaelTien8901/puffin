## Why

The Puffin tutorial site scores 4/10 for beginner friendliness. Content leads with technical jargon (e.g., "excess returns above a benchmark") rather than intuitive explanations. Users without a finance background hit a steep learning curve in the first few pages and may abandon the guide.

## What Changes

- Add "Plain English" callout boxes at the start of key chapters that explain financial concepts using everyday analogies before diving into technical definitions
- Add a conceptual overview (1-2 paragraphs) at the top of each major Part's landing page to ground readers before they encounter jargon
- Target three high-impact sections first: Part 1 (Market Foundations), Part 4 (Alpha Factors), and Part 24 (Risk Management)
- Use Just the Docs `{: .note }` callout blocks for consistent styling

## Capabilities

### New Capabilities
- `concept-callouts`: Plain-English analogy callouts inserted into chapter content for key financial terms (Order Book, Market Maker, Alpha, Beta, Sharpe Ratio, Drawdown, Liquidity, Momentum)
- `conceptual-overviews`: Introductory plain-English overview paragraphs on landing pages for Parts 1, 4, and 24

### Modified Capabilities
_(none — no existing specs)_

## Impact

- **Files modified**: Article pages in `docs/01-market-foundations/`, `docs/04-alpha-factors/`, `docs/24-risk-management/` and their `index.md` landing pages
- **No code changes** — content-only modifications to tutorial markdown files
- **No dependencies** — uses existing Just the Docs callout infrastructure
