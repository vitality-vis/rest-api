-- One research-notes document per signed-in user.
create table public.user_notes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  content text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

create index user_notes_user_updated_idx
  on public.user_notes (user_id, updated_at desc);

create or replace function public.touch_user_notes_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger user_notes_touch_updated_at
before update on public.user_notes
for each row
execute function public.touch_user_notes_updated_at();

-- Flask uses the service role for server-side queries. Grants and RLS still
-- protect the table if an authenticated user token reaches it directly.
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.user_notes to authenticated;

alter table public.user_notes enable row level security;

create policy user_notes_manage_own
  on public.user_notes for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
