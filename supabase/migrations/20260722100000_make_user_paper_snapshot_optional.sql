-- DOI-backed papers are rehydrated from the canonical catalog, so a local
-- metadata snapshot is only required for papers without a DOI.
alter table public.user_papers
  alter column metadata_snapshot drop not null;
