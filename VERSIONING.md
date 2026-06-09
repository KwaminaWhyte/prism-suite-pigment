# Versioning

This repo follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

While pre-1.0 (`0.y.z`):

- **minor** (`0.Y`) — new features or notable behaviour changes,
- **patch** (`0.y.Z`) — fixes and small tweaks.

1.0 stability guarantees do not apply yet; APIs and file formats may change
between minor versions. File formats stay backward-readable wherever practical
(new fields are added with `#[serde(default)]`).

## Where the version lives

`Cargo.toml` → `[workspace.package] version`. Every crate inherits it via
`version.workspace = true`.

## Cutting a release

1. Move the `## [Unreleased]` block of `CHANGELOG.md` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section; start a fresh empty `## [Unreleased]`.
2. Bump `[workspace.package] version` in `Cargo.toml`.
3. Commit: `chore(release): vX.Y.Z`.
4. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.

Changelog format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
