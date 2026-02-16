## ADDED Requirements

### Requirement: Static site generation with Jekyll and Just the Docs
The system SHALL use Jekyll with the Just the Docs theme to generate a static tutorial site deployable to GitHub Pages, consistent with the existing Rust guide tutorial project.

#### Scenario: Site builds successfully
- **WHEN** the user runs `bundle exec jekyll build`
- **THEN** a static site is generated in the `_site/` directory with all tutorial content rendered as HTML

#### Scenario: Local development preview via Docker
- **WHEN** the user runs `docker-compose up` in the `docs/` directory
- **THEN** a local Jekyll development server starts with live-reload on content changes

### Requirement: GitHub Pages deployment
The system SHALL deploy the tutorial site to GitHub Pages using GitHub's native Jekyll support or GitHub Actions.

#### Scenario: Auto-deploy on push to main
- **WHEN** a commit is pushed to the `main` branch
- **THEN** GitHub Pages builds and deploys the site automatically

#### Scenario: Build verification for pull requests
- **WHEN** a pull request is opened with documentation changes
- **THEN** the CI pipeline builds the site and reports success/failure

### Requirement: Progressive module navigation
The system SHALL organize tutorial content into numbered parts with clear navigation between chapters, using Just the Docs navigation structure.

#### Scenario: Module ordering
- **WHEN** a learner visits the tutorial site
- **THEN** modules are displayed in order (01-market-foundations through 09-monitoring-analytics) in the sidebar navigation

#### Scenario: Chapter navigation
- **WHEN** a learner finishes a chapter
- **THEN** next/previous navigation links are available to move between chapters

### Requirement: Code snippet integration
The system SHALL embed code examples in tutorial pages using Jekyll includes or fenced code blocks with Rouge syntax highlighting.

#### Scenario: Syntax-highlighted code
- **WHEN** a tutorial page contains a Python code block
- **THEN** it is rendered with syntax highlighting via Rouge

### Requirement: Search functionality
The system SHALL provide full-text search across all tutorial content using Just the Docs built-in search.

#### Scenario: Search returns relevant results
- **WHEN** a learner searches for "moving average"
- **THEN** all tutorial pages containing that term are returned as search results

### Requirement: Callout blocks for tips, warnings, and notes
The system SHALL use Just the Docs callout blocks to highlight important information, warnings, and tips throughout tutorials.

#### Scenario: Callout rendering
- **WHEN** a tutorial page uses a callout block (note, tip, warning, important)
- **THEN** it renders with the appropriate color and icon matching the callout type

### Requirement: Mermaid diagram support
The system SHALL support Mermaid diagrams for visualizing architectures, data flows, and trading system concepts.

#### Scenario: Diagram rendering
- **WHEN** a tutorial page contains a Mermaid code block
- **THEN** it renders as an interactive diagram in the browser
