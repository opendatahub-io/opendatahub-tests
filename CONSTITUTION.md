# OpenDataHub-Tests Constitution

This constitution defines the non-negotiable principles and governance rules for the opendatahub-tests repository. It applies to all test development, whether performed by humans or AI assistants.

## Core Principles

### I. Simplicity First

All changes MUST favor the simplest solution that works. Complexity MUST be justified.

- Aim for the simplest solution that works while keeping the code clean
- Do not prepare code for the future just because it may be useful (YAGNI)
- Every function, variable, fixture, and test written MUST be used, or else removed
- Flexible code MUST NOT come at the expense of readability

**Rationale**: The codebase is maintained by multiple teams; simplicity ensures maintainability and reduces cognitive load.

### II. Code Consistency

All changes MUST follow existing code patterns and architecture.

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use pre-commit hooks to enforce style (ruff, mypy, flake8)
- Use absolute import paths; import specific functions rather than modules
- Use descriptive names; meaningful names are better than short names
- Add type annotations to all new code; follow the rules defined in [pyproject.toml](./pyproject.toml)

**Rationale**: Consistent patterns reduce the learning curve and prevent architectural drift.

### III. Test Clarity and Dependencies

Each test MUST verify a single aspect of the product and may be dependent on other tests.

- Tests MUST have a clear purpose and be easy to understand
- Tests MUST be properly documented with docstrings explaining what the test does
- When test dependencies exist, use pytest-dependency plugin to declare them explicitly, encourage use of dependency marker(s) when possible
- Group related tests in classes only when they share fixtures; never group unrelated tests

### IV. Fixture Discipline

Fixtures MUST do one thing only and follow proper scoping.

- Fixture names MUST be nouns describing what they provide (e.g., `storage_secret` not `create_secret`)
- Fixtures MUST handle setup and teardown using context managers where appropriate
- Use the narrowest fixture scope that meets the need (function > class > module > session)
- Conftest.py files MUST contain fixtures only; no utility functions or constants
- Use `request.param` with dict structures for parameterized fixtures

**Rationale**: Single-responsibility fixtures are easier to debug, reuse, and compose.

### V. Interacting with Kubernetes Resources

All cluster interactions MUST use openshift-python-wrapper or oc CLI.

- Use [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper) for all K8s API calls
- For missing resources, generate them using class_generator and contribute to wrapper
- Use oc CLI only when wrapper is not relevant (e.g., must-gather generation)
- Resource lifecycle MUST be managed via context managers to ensure cleanup

**Rationale**: Consistent API abstraction ensures portability between ODH (upstream) and RHOAI (downstream).

### VI. Locality of Behavior

Keep code close to where it is used.

- Keep functions and fixtures close to where they're used initially
- Move to shared locations (utilities, common conftest) only when multiple modules need them
- Avoid creating abstractions prematurely
- Small, focused changes are preferred unless explicitly asked otherwise

**Rationale**: Locality reduces navigation overhead and makes the impact of changes obvious.

### VII. Security Awareness

All code MUST consider security implications.

- Never log/expose secrets; redact/mask if printing is unavoidable
- Avoid running destructive commands without explicit user confirmation
- Use detect-secrets and gitleaks pre-commit hooks to prevent secret leakage
- Test code MUST NOT introduce vulnerabilities into the tested systems

**Rationale**: Tests interact with production-like clusters; security lapses can have real consequences.

## Test Development Standards

### Test Documentation

- Every test or test class MUST have a docstring explaining what it tests
- Docstrings MUST be understandable by engineers from other components, managers, or PMs
- Use Google-format docstrings
- Comments are allowed only for complex code blocks (e.g., complex regex)

### Test Markers

- All tests MUST apply relevant markers from pytest.ini
- Use tier markers (smoke, sanity, tier1, tier2) to indicate test priority
- Use component markers (model_explainability, llama_stack, rag) for ownership
- Use infrastructure markers (gpu, parallel, slow) for execution filtering

### Test Organization

- Tests are organized by component in `tests/<component>/`
- Each component has its own conftest.py for scoped fixtures
- Utilities go in `utilities/` with topic-specific modules

## AI-Assisted Development Guidelines

### Developer Responsibility

Developers are ultimately responsible for all code, regardless of whether AI tools assisted.

- Always assume AI-generated code is unsafe and incorrect until verified
- Double-check all AI suggestions against project patterns and this constitution
- AI tools MUST be guided by AGENTS.md (symlink to CLAUDE.md if needed)

### AI Code Generation Rules

- AI MUST follow existing patterns; never introduce new architectural concepts without justification
- AI MUST NOT add unnecessary complexity or "helpful" abstractions
- AI-generated tests MUST have proper docstrings and markers
- AI MUST ask when in doubt about requirements or patterns

### Specification-Driven Development

When adopting AI-driven spec development:

- Specifications MUST be in structured format (YAML/JSON with defined schema)
- Tests MUST include requirement traceability (Polarion, Jira markers)
- Docstrings MUST follow Given-When-Then pattern for behavioral clarity
- Generated tests MUST pass pre-commit checks before review

## Governance

### Constitution Authority

This constitution supersedes all other practices when there is a conflict. All PRs and reviews MUST verify compliance.

### Amendment Process

1. Propose changes via PR to `CONSTITUTION.md`
2. Changes require review by at least two maintainers
3. Breaking changes (principle removal/redefinition) require team discussion

### Versioning Policy

No versioning policy is enforced.

### Compliance Review

- All PRs MUST be verified against constitution principles
- Pre-commit hooks enforce code quality standards
- CI (tox) validates test structure and typing
- Two reviewers required; verified label required before merge

### Guidance Reference

For development runtime guidance, consult:
- [AGENTS.md](./AGENTS.md) for AI assistant instructions
- [DEVELOPER_GUIDE.md](./docs/DEVELOPER_GUIDE.md) for contribution details
- [STYLE_GUIDE.md](./docs/STYLE_GUIDE.md) for code style

**Version**: 1.0.0 | **Ratified**: 2026-01-08 | **Last Amended**: 2026-01-08
