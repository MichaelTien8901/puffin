## ADDED Requirements

### Requirement: Every chapter index SHALL have a Related Chapters section
Each chapter's `index.md` SHALL include a `## Related Chapters` section near the bottom (before Source Code) with 2–4 bullet points linking to related chapters.

#### Scenario: Related Chapters section exists
- **WHEN** a chapter's `index.md` is rendered
- **THEN** it SHALL contain a `## Related Chapters` heading with 2–4 bulleted links using `{{ site.baseurl }}` URLs

#### Scenario: Links follow cluster relationships
- **WHEN** the Related Chapters section is authored
- **THEN** links SHALL connect chapters that share a logical relationship (e.g., prerequisite, builds-upon, uses-output-of, alternative-approach)

### Requirement: Sub-pages SHALL include inline cross-references where relevant
When a sub-page references a concept covered in depth by another chapter, it SHALL include an inline link to that chapter.

#### Scenario: Inline cross-reference
- **WHEN** a sub-page mentions a concept explained in another chapter (e.g., "alpha factors" mentioned in the tree ensembles chapter)
- **THEN** the first mention SHALL include a hyperlink to the relevant chapter

### Requirement: Cross-links SHALL be bidirectional
If chapter A links to chapter B in its Related Chapters section, chapter B SHALL link back to chapter A.

#### Scenario: Bidirectional linking
- **WHEN** chapter 04 (alpha factors) lists chapter 11 (tree ensembles) as related
- **THEN** chapter 11 SHALL also list chapter 04 as related
