## ADDED Requirements

### Requirement: Every chapter SHALL include working puffin module imports
Each chapter SHALL contain at least one Python code block that imports from the `puffin` package and demonstrates actual API usage, not pseudocode.

#### Scenario: Code example with real imports
- **WHEN** a reader views a chapter's code examples
- **THEN** at least one code block SHALL use `from puffin.xxx import YYY` with a realistic usage example

#### Scenario: Code examples match current API
- **WHEN** a code example references a puffin module
- **THEN** the import paths and function signatures SHALL match the actual `puffin/` package implementation

### Requirement: Code examples SHALL follow the theory-then-code pattern
Code examples SHALL appear after the theoretical explanation they demonstrate, following the chapter template convention.

#### Scenario: Code placement
- **WHEN** a concept is explained in prose
- **THEN** a corresponding code example SHALL follow within the same sub-page, demonstrating the concept with puffin modules

### Requirement: Code examples SHALL be self-contained within context
Each code block SHALL include enough context (imports, sample data setup) that a reader can understand it without scrolling to other pages.

#### Scenario: Self-contained example
- **WHEN** a code block is read in isolation
- **THEN** it SHALL include all necessary imports and any data setup needed to understand the example
