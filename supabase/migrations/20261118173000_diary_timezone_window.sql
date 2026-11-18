-- Accept every time zone's local "yesterday" and "today".
-- Local dates span UTC-2 .. UTC+1 days relative to the UTC date; the UI enforces the exact rule.
begin;

create or replace function public.complete_diary_day(target_day date)
returns date
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  utc_today date := (now() at time zone 'utc')::date;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if target_day is null or target_day < utc_today - 2 or target_day > utc_today + 1 then
    raise exception 'Only yesterday and today can be completed';
  end if;
  insert into public.diary_completions(user_id, entry_day)
  values (participant, target_day)
  on conflict do nothing;
  return target_day;
end;
$$;

create or replace function public.uncomplete_diary_day(target_day date)
returns date
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  utc_today date := (now() at time zone 'utc')::date;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if target_day is null or target_day < utc_today - 2 or target_day > utc_today + 1 then
    raise exception 'Only yesterday and today can be changed';
  end if;
  delete from public.diary_completions
  where user_id = participant and entry_day = target_day;
  return target_day;
end;
$$;

commit;
