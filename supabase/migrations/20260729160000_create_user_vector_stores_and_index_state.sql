-- One private Azure Vector Store per user; Files remain the authoritative PDF record.
create table public.user_vector_stores (
  user_id uuid primary key references auth.users (id) on delete cascade,
  azure_vector_store_id text not null unique,
  status text not null default 'ready' check (status in ('creating', 'ready', 'error')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger user_vector_stores_touch_updated_at
before update on public.user_vector_stores
for each row
execute function public.touch_user_papers_updated_at();

grant select, insert, update, delete on public.user_vector_stores to authenticated;
alter table public.user_vector_stores enable row level security;
create policy user_vector_stores_manage_own
  on public.user_vector_stores for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

alter table public.user_papers
  add column vs_file_status text not null default 'not_indexed'
    check (vs_file_status in ('not_indexed', 'pending', 'in_progress', 'completed', 'failed')),
  add column vs_file_id text,
  add column vs_indexed_at timestamptz,
  add column vs_last_error text;

create index user_papers_user_vs_status_idx
  on public.user_papers (user_id, vs_file_status);
