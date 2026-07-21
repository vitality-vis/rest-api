-- A signed-in user's saved-paper shelf. The snapshot is a display fallback;
-- the main paper dataset remains the authoritative metadata source.
create table public.user_papers (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  paper_id text not null,
  metadata_snapshot jsonb not null,

  azure_file_id text unique,
  uploaded_filename text,
  uploaded_bytes bigint,
  uploaded_at timestamptz,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, paper_id)
);

create index user_papers_user_created_idx
  on public.user_papers (user_id, created_at desc);

create or replace function public.touch_user_papers_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger user_papers_touch_updated_at
before update on public.user_papers
for each row
execute function public.touch_user_papers_updated_at();

-- Flask uses the service role for server-side queries. The grants and RLS are
-- still required protection if an authenticated user token reaches this table.
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.user_papers to authenticated;

alter table public.user_papers enable row level security;

create policy user_papers_manage_own
  on public.user_papers for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
