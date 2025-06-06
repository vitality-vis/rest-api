
def bib_template(paper):
    title = paper.get('Title', '')
    year = paper.get('Year', '')
    authors = []
    for author in paper.get('Authors', []):
        name = author.split(' ')
        authors.append(f'{name[1]}, {name[0]}')
    bib_id = f"{authors[0].split(',')[0]}{year}{title.split(' ')[0]}"

    return f'''
@article{{{bib_id},
  title={{{title}}},
  author={{{' and '.join(authors)}}},
  year={{{year}}}
}}
'''