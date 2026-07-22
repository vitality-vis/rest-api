-- Saved and full-text upload are independent states on user_papers.
-- Existing rows were created by the save-only shelf API, so backfill them as saved.
-- Guard the backfill so re-applying this migration cannot mark upload-only rows as saved.
do $$
begin
  if not exists (
    select 1
    from information_schema.columns
    where table_schema = 'public'
      and table_name = 'user_papers'
      and column_name = 'is_saved'
  ) then
    alter table public.user_papers
      add column is_saved boolean not null default false;

    update public.user_papers
    set is_saved = true;
  end if;
end $$;

create index if not exists user_papers_user_saved_created_idx
  on public.user_papers (user_id, created_at desc)
  where is_saved = true;
