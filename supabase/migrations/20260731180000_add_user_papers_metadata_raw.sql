-- Preserve the source payload for user-imported papers without changing the
-- canonical metadata_snapshot consumed by the application.
alter table public.user_papers
  add column if not exists metadata_raw jsonb;

comment on column public.user_papers.metadata_raw is
  'Original user-provided import payload; not returned by normal shelf reads.';
