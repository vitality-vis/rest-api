-- Paper identity on the personal shelf: corpus (Vitality main dataset) vs user (imported).
-- Existing rows were created only by catalog save/upload paths → backfill as corpus.

alter table public.user_papers
  add column if not exists origin text;

update public.user_papers
set origin = 'corpus'
where origin is null;

alter table public.user_papers
  alter column origin set default 'corpus',
  alter column origin set not null;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'user_papers_origin_check'
      and conrelid = 'public.user_papers'::regclass
  ) then
    alter table public.user_papers
      add constraint user_papers_origin_check
      check (origin in ('corpus', 'user'));
  end if;
end $$;

create index if not exists user_papers_user_origin_idx
  on public.user_papers (user_id, origin);
