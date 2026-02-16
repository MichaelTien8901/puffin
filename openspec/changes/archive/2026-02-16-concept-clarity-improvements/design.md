## Context

The Puffin tutorial articles define financial concepts using textbook language. Readers without finance backgrounds encounter jargon (Alpha, Sharpe Ratio, Order Book) before understanding the intuition. Three sections are highest priority: Part 1 (Market Foundations), Part 4 (Alpha Factors), Part 24 (Risk Management).

## Goals / Non-Goals

**Goals:**
- Add plain-English analogies for key financial terms inside existing chapter content
- Add conceptual overview paragraphs to Part 1, 4, and 24 landing pages
- Use existing Just the Docs callout syntax — no new infrastructure

**Non-Goals:**
- Rewriting or simplifying the technical content itself
- Adding analogies to every chapter across all 25 Parts
- Creating a standalone glossary page (may be a future change)
- Modifying code examples or Python modules

## Decisions

**Callout format**: Use Just the Docs `{: .tip }` blocks with a "Plain English" header. This visually separates analogies from technical content so advanced readers can skip them.
- Alternative considered: Inline parenthetical explanations — rejected because they break reading flow and are harder to scan.

**Placement**: Insert callout immediately after the first technical use of each term, not at the top of the page. This keeps context close to where the jargon appears.
- Alternative considered: Glossary sidebar or top-of-page summary — rejected because terms are better understood in context.

**Scope**: Target ~8 key terms across 3 sections (Order Book, Market Maker, Liquidity, Alpha, Beta, Momentum, Sharpe Ratio, Drawdown). This covers the highest-friction concepts without bloating every page.

**Landing page overviews**: Add 1-2 plain-English paragraphs at the top of each Part's `index.md`, before the chapter listing. Frame what the section teaches in practical terms.

## Risks / Trade-offs

- [Analogies oversimplify] → Each callout includes "In technical terms:" link back to the full definition nearby
- [Content drift] → Analogies are self-contained callout blocks, easy to update independently of technical content
- [Cluttered pages] → Limited to ~2-3 callouts per article page; only the most jargon-heavy terms get treatment
