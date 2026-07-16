"""Shared helpers (e.g. BibTeX formatting) for the research assistant."""


def bib_template(paper: dict) -> str:
    title = paper.get('Title', '')
    year = paper.get('Year', '')
    authors = []
    for author in paper.get("Authors", []):
        parts = author.split(" ", 1)
        if len(parts) >= 2:
            authors.append(f"{parts[1]}, {parts[0]}")
        elif parts:
            authors.append(parts[0])
    first_author = authors[0].split(",")[0] if authors else "unknown"
    first_word = title.split()[0] if title.split() else "article"
    bib_id = f"{first_author}{year}{first_word}"

    return f'''
@article{{{bib_id},
  title={{{title}}},
  author={{{' and '.join(authors)}}},
  year={{{year}}}
}}
'''