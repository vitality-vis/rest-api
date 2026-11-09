-- Minimal, authenticated context query for latency-sensitive note completions.
create function public.get_note_completion_context(p_project_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  current_user_id uuid := auth.uid();
  result jsonb;
begin
  if current_user_id is null then
    raise exception 'Authentication required' using errcode = '28000';
  end if;

  if not exists (
    select 1
    from public.projects as projects
    where projects.id = p_project_id
      and projects.user_id = current_user_id
  ) then
    raise exception 'Project not found' using errcode = 'P0002';
  end if;

  select jsonb_build_object(
    'saved_total', count(*),
    'papers', coalesce(
      jsonb_agg(
        jsonb_build_object(
          'paper_id', candidates.paper_id,
          'title', candidates.title
        )
        order by candidates.membership_created_at, candidates.user_paper_id
      ),
      '[]'::jsonb
    )
  )
  into result
  from (
    select
      papers.id as user_paper_id,
      papers.paper_id,
      left(
        btrim(coalesce(
          papers.metadata_snapshot ->> 'Title',
          papers.metadata_snapshot ->> 'title'
        )),
        300
      ) as title,
      memberships.created_at as membership_created_at
    from public.project_papers as memberships
    join public.user_papers as papers
      on papers.id = memberships.user_paper_id
      and papers.user_id = memberships.user_id
    where memberships.project_id = p_project_id
      and memberships.user_id = current_user_id
      and nullif(btrim(coalesce(
        papers.metadata_snapshot ->> 'Title',
        papers.metadata_snapshot ->> 'title'
      )), '') is not null
  ) as candidates;

  return result;
end;
$$;

revoke all on function public.get_note_completion_context(uuid)
  from public, anon, authenticated;
grant execute on function public.get_note_completion_context(uuid)
  to authenticated;
