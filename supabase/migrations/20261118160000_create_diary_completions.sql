-- Track which days a participant completed the daily diary (Google Form).
-- The survey itself lives outside the app; this only records "done" per day.
begin;

create table public.diary_completions (
  user_id uuid not null references auth.users(id),
  entry_day date not null,
  completed_at timestamptz not null default now(),
  primary key (user_id, entry_day)
);

alter table public.diary_completions enable row level security;
revoke all on public.diary_completions from anon, authenticated;
grant select on public.diary_completions to authenticated;
create policy diary_read_own on public.diary_completions for select to authenticated
  using (user_id = (select auth.uid()));

-- Only "yesterday" and "today" can be completed. The client sends its local date, so allow
-- a +/-1 day window around the UTC date to cover every time zone; the UI enforces the exact rule.
create function public.complete_diary_day(target_day date)
returns date
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  utc_today date := (now() at time zone 'utc')::date;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if target_day is null or target_day < utc_today - 1 or target_day > utc_today + 1 then
    raise exception 'Only yesterday and today can be completed';
  end if;
  insert into public.diary_completions(user_id, entry_day)
  values (participant, target_day)
  on conflict do nothing;
  return target_day;
end;
$$;

revoke all on function public.complete_diary_day(date) from public, anon;
grant execute on function public.complete_diary_day(date) to authenticated;

commit;
