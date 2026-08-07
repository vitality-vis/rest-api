-- User-defined labels for papers on the Saved shelf. Tags remain on the
-- library row so catalog metadata refreshes cannot overwrite them.
alter table public.user_papers
  add column if not exists tags text[] not null default '{}';

comment on column public.user_papers.tags is
  'User-defined tags for a saved paper; cleared when the paper is unsaved.';
