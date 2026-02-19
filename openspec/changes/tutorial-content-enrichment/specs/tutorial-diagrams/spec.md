## ADDED Requirements

### Requirement: Every chapter SHALL have at least one Mermaid diagram
Each chapter's `index.md` SHALL include at least one Mermaid diagram providing a visual overview of the chapter's architecture, data flow, or decision logic.

#### Scenario: Index page diagram
- **WHEN** a chapter's `index.md` is rendered
- **THEN** it SHALL contain at least one Mermaid code block (```mermaid)

#### Scenario: Sub-page diagrams for complex topics
- **WHEN** a sub-page explains an algorithm, pipeline, or multi-step process
- **THEN** it SHOULD include a focused Mermaid diagram illustrating that specific flow

### Requirement: Diagrams SHALL follow the project's dark-fill styling convention
All Mermaid diagrams SHALL use the styling convention documented in CLAUDE.md: dark fill colors (RGB channels ≤ 0x90), paper background (`#d5d0c8`), dark text (`#2c3e50`), and edges in `#4a5568`.

#### Scenario: Diagram styling compliance
- **WHEN** a Mermaid diagram is rendered
- **THEN** node fills SHALL use RGB values where each channel ≤ 0x90 (e.g., `#2d5016`, `#1a3a5c`)
- **AND** the diagram background SHALL be `#d5d0c8`
- **AND** text color SHALL be `#e8e0d4` on dark fills or `#2c3e50` on labels

### Requirement: Diagrams SHALL be kept simple
Each diagram SHALL have no more than 15 nodes to ensure readability and fast rendering.

#### Scenario: Diagram complexity limit
- **WHEN** a Mermaid diagram is authored
- **THEN** it SHALL contain 15 or fewer nodes
