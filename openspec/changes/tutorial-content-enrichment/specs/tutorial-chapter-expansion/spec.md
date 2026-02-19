## ADDED Requirements

### Requirement: Single-file chapters SHALL be split into multi-page sub-chapters
Each of the 22 single-file tutorial chapters SHALL be converted into a multi-page structure with an `index.md` parent page and 2–4 numbered child pages (`01-subtopic.md`, `02-subtopic.md`, etc.).

#### Scenario: Chapter folder structure after expansion
- **WHEN** any tutorial chapter folder is inspected
- **THEN** it SHALL contain an `index.md` and at least 2 numbered child `.md` files

#### Scenario: Existing content is preserved
- **WHEN** a single-file chapter is split into sub-pages
- **THEN** all existing prose, code examples, tables, callouts, and exercises from the original file SHALL appear in the resulting sub-pages

### Requirement: Index pages SHALL use Just the Docs parent navigation
Each `index.md` SHALL have `has_children: true` in its front matter, and each child page SHALL have a `parent:` field matching the index page's `title:`.

#### Scenario: Index page front matter
- **WHEN** a chapter `index.md` is rendered
- **THEN** its front matter SHALL include `layout: default`, `title:`, `nav_order:`, `has_children: true`, and `permalink:`

#### Scenario: Child page front matter
- **WHEN** a child page is rendered
- **THEN** its front matter SHALL include `layout: default`, `title:`, `parent:` matching the index title, and `nav_order:` for ordering within the chapter

### Requirement: Existing permalinks SHALL be preserved
The `index.md` for each chapter SHALL use the same `permalink:` as the original single-file page to avoid broken links.

#### Scenario: Permalink continuity
- **WHEN** a reader visits an existing chapter URL (e.g., `/07-backtesting/`)
- **THEN** the request SHALL resolve to the new `index.md` page without a 404

### Requirement: Sub-page splits SHALL follow natural topic boundaries
Chapters SHALL be split along existing section boundaries (e.g., theory, implementation, advanced usage, exercises) rather than arbitrary line counts.

#### Scenario: Logical grouping
- **WHEN** a chapter with sections on theory, implementation, and advanced usage is split
- **THEN** each sub-page SHALL cover a cohesive topic that can be read independently
