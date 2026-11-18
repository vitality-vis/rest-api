-- Let participants correct an accidental completion for yesterday or today.
begin;

create function public.uncomplete_diary_day(target_day date)
returns date
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  utc_today date := (now() at time zone 'utc')::date;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if target_day is null or target_day < utc_today - 1 or target_day > utc_today + 1 then
    raise exception 'Only yesterday and today can be changed';
  end if;
  delete from public.diary_completions
  where user_id = participant and entry_day = target_day;
  return target_day;
end;
$$;

revoke all on function public.uncomplete_diary_day(date) from public, anon;
grant execute on function public.uncomplete_diary_day(date) to authenticated;

commit;
