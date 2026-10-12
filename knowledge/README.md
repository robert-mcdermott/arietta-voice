# Knowledge

Add user-editable markdown articles to this folder.

Only this README and `getting-started.md` are tracked by default. All other
files and subdirectories here are ignored by Git, so add personal knowledge
in new files rather than editing the tracked starter article.

The article template lives at
[`examples/knowledge-template.md`](../examples/knowledge-template.md). Copy it
into this directory with a descriptive name to create your own article.

Git ignore rules do not protect files that were already tracked or added with
`git add -f`. Before committing, inspect `git diff --cached --name-only`.
If your own articles were previously tracked, use `git rm --cached -- path`
for those specific files to retain the local copies while stopping tracking;
this does not remove content from earlier commits. A knowledge directory
configured elsewhere needs its own ignore rules or a location outside the repo.

Recommended structure:

- one topic per file
- short YAML frontmatter with `title`, `tags`, and optional `aliases`
- clear headings for individual sections
- short, factual paragraphs instead of giant walls of text

The assistant can search these files with the `keyword`, `vector`, or `hybrid` backends.

`getting-started.md` is a generic example knowledge article.
