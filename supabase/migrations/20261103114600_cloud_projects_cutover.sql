-- Downtime-only cloud Projects cutover.
-- All DDL, data movement, verification, and contraction commit atomically.
begin;

-- Project ownership foundation. Authenticated clients may read their projects,
-- but mutations remain behind controlled RPCs so default-project invariants
-- cannot be bypassed by direct table writes.
create table public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text,
  is_default boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint projects_id_user_id_key unique (id, user_id),
  constraint projects_name_nonempty check (btrim(name) <> '')
);

create unique index projects_one_default_per_user_idx
  on public.projects (user_id)
  where is_default = true;

create index projects_user_updated_idx
  on public.projects (user_id, updated_at desc);

create or replace function public.touch_projects_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger projects_touch_updated_at
before update on public.projects
for each row
execute function public.touch_projects_updated_at();

revoke all on function public.touch_projects_updated_at() from public, anon, authenticated;

grant usage on schema public to authenticated;
revoke insert, update, delete on public.projects from public, anon, authenticated;
grant select on public.projects to authenticated;

alter table public.projects enable row level security;

create policy projects_read_own
  on public.projects for select to authenticated
  using (user_id = (select auth.uid()));

-- Seed one default for each existing user. The inference predicate matches the
-- partial unique index, making this safe to retry or race with another creator.
insert into public.projects (user_id, name, is_default)
select users.id, 'My Project', true
from auth.users as users
where not exists (
  select 1
  from public.projects as projects
  where projects.user_id = users.id
    and projects.is_default = true
)
on conflict (user_id) where is_default = true do nothing;

create function public.get_or_create_default_project()
returns setof public.projects
language plpgsql
security definer
set search_path = ''
as $$
declare
  current_user_id uuid := auth.uid();
begin
  if current_user_id is null then
    raise exception 'Authentication required' using errcode = '28000';
  end if;

  -- Serialize default creation per user so concurrent first requests converge
  -- on one row before attempting the partial-unique-index insert.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(current_user_id::text, 0)
  );

  return query
  select projects.*
  from public.projects as projects
  where projects.user_id = current_user_id
    and projects.is_default = true;

  if found then
    return;
  end if;

  insert into public.projects (user_id, name, is_default)
  values (current_user_id, 'My Project', true)
  on conflict (user_id) where is_default = true do nothing;

  return query
  select projects.*
  from public.projects as projects
  where projects.user_id = current_user_id
    and projects.is_default = true;
end;
$$;

revoke all on function public.get_or_create_default_project() from public, anon, authenticated;
grant execute on function public.get_or_create_default_project() to authenticated, service_role;

-- The redundant ownership key supports later tenant-safe composite foreign keys.
alter table public.user_papers
  add constraint user_papers_id_user_id_key unique (id, user_id);

-- Controlled project lifecycle operations. Ownership always comes from auth.uid().

alter table public.projects
  add constraint projects_name_max_length check (char_length(btrim(name)) <= 120);

create function public.create_project(
  p_name text,
  p_description text default null
)
returns public.projects
language plpgsql
security definer
set search_path = ''
as $$
declare
  current_user_id uuid := auth.uid();
  normalized_name text := btrim(p_name);
  created_project public.projects;
begin
  if current_user_id is null then
    raise exception 'Authentication required' using errcode = '28000';
  end if;
  if normalized_name is null or normalized_name = '' then
    raise exception 'Project name is required' using errcode = '22023';
  end if;
  if char_length(normalized_name) > 120 then
    raise exception 'Project name must be at most 120 characters' using errcode = '22023';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(current_user_id::text, 0)
  );
  perform public.get_or_create_default_project();

  insert into public.projects (user_id, name, description, is_default)
  values (current_user_id, normalized_name, p_description, false)
  returning * into created_project;
  return created_project;
end;
$$;

create function public.rename_project(p_project_id uuid, p_name text)
returns public.projects
language plpgsql
security definer
set search_path = ''
as $$
declare
  current_user_id uuid := auth.uid();
  normalized_name text := btrim(p_name);
  renamed_project public.projects;
begin
  if current_user_id is null then
    raise exception 'Authentication required' using errcode = '28000';
  end if;
  if normalized_name is null or normalized_name = '' then
    raise exception 'Project name is required' using errcode = '22023';
  end if;
  if char_length(normalized_name) > 120 then
    raise exception 'Project name must be at most 120 characters' using errcode = '22023';
  end if;

  update public.projects
  set name = normalized_name
  where id = p_project_id
    and user_id = current_user_id
  returning * into renamed_project;

  if renamed_project.id is null then
    raise exception 'Project not found' using errcode = 'P0002';
  end if;
  return renamed_project;
end;
$$;

revoke all on function public.create_project(text, text) from public, anon, authenticated;
revoke all on function public.rename_project(uuid, text) from public, anon, authenticated;
grant execute on function public.create_project(text, text) to authenticated;
grant execute on function public.rename_project(uuid, text) to authenticated;

create table public.project_papers (
  user_id uuid not null,
  project_id uuid not null,
  user_paper_id uuid not null,
  tags text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint project_papers_pkey primary key (project_id, user_paper_id),
  constraint project_papers_project_owner_fkey
    foreign key (project_id, user_id) references public.projects (id, user_id) on delete cascade,
  constraint project_papers_user_paper_owner_fkey
    foreign key (user_paper_id, user_id) references public.user_papers (id, user_id) on delete restrict
);

create index project_papers_user_asset_idx on public.project_papers (user_id, user_paper_id);
create index project_papers_user_project_created_idx
  on public.project_papers (user_id, project_id, created_at desc);

create function public.touch_project_papers_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger project_papers_touch_updated_at
before update on public.project_papers
for each row execute function public.touch_project_papers_updated_at();

revoke all on function public.touch_project_papers_updated_at() from public, anon, authenticated;
revoke insert, update, delete on public.project_papers from public, anon, authenticated;
grant select on public.project_papers to authenticated;
alter table public.project_papers enable row level security;

alter table public.user_notes add column project_id uuid;
alter table public.user_notes add constraint user_notes_project_owner_fkey
  foreign key (project_id, user_id) references public.projects (id, user_id) on delete restrict;
create index user_notes_user_project_updated_idx
  on public.user_notes (user_id, project_id, updated_at desc);

alter table public.chat_conversations add column project_id uuid;
alter table public.chat_conversations add constraint chat_conversations_project_owner_fkey
  foreign key (project_id, user_id) references public.projects (id, user_id) on delete restrict;
create index chat_conversations_user_project_updated_idx
  on public.chat_conversations (user_id, project_id, updated_at desc);

-- Assign all existing project-scoped data to each owner's default project.
update public.user_notes as notes
set project_id = defaults.id
from public.projects as defaults
where defaults.user_id = notes.user_id
  and defaults.is_default = true;

update public.chat_conversations as conversations
set project_id = defaults.id
from public.projects as defaults
where defaults.user_id = conversations.user_id
  and defaults.is_default = true;

-- Only legacy saved assets become shelf memberships; preserve their tags.
insert into public.project_papers (user_id, project_id, user_paper_id, tags)
select papers.user_id, defaults.id, papers.id, coalesce(papers.tags, '{}')
from public.user_papers as papers
join public.projects as defaults
  on defaults.user_id = papers.user_id and defaults.is_default = true
where papers.is_saved = true
on conflict (project_id, user_paper_id) do nothing;

-- Abort the whole migration if any row cannot satisfy the final ownership model.
do $$
begin
  if exists (select 1 from public.user_notes where project_id is null) then
    raise exception 'Cutover verification failed: user_notes.project_id remains null';
  end if;
  if exists (select 1 from public.chat_conversations where project_id is null) then
    raise exception 'Cutover verification failed: chat_conversations.project_id remains null';
  end if;
  if exists (
    select 1 from public.user_notes n
    left join public.projects p on p.id = n.project_id and p.user_id = n.user_id
    where p.id is null
  ) or exists (
    select 1 from public.chat_conversations c
    left join public.projects p on p.id = c.project_id and p.user_id = c.user_id
    where p.id is null
  ) then
    raise exception 'Cutover verification failed: note or conversation ownership mismatch';
  end if;
  if exists (
    select 1 from public.user_papers papers
    join public.projects defaults on defaults.user_id = papers.user_id and defaults.is_default = true
    where papers.is_saved = true and not exists (
      select 1 from public.project_papers memberships
      where memberships.project_id = defaults.id
        and memberships.user_paper_id = papers.id
        and memberships.user_id = papers.user_id
    )
  ) then
    raise exception 'Cutover verification failed: saved paper lacks default membership';
  end if;
  if exists (
    select 1 from public.project_papers memberships
    left join public.projects projects
      on projects.id = memberships.project_id and projects.user_id = memberships.user_id
    left join public.user_papers papers
      on papers.id = memberships.user_paper_id and papers.user_id = memberships.user_id
    where projects.id is null or papers.id is null
  ) then
    raise exception 'Cutover verification failed: project membership ownership mismatch';
  end if;
end;
$$;

-- Legacy shelf fields have served their only purpose in the backfill and checks above.
alter table public.user_papers drop column is_saved;
alter table public.user_papers drop column tags;

alter table public.user_notes alter column project_id set not null;
alter table public.chat_conversations alter column project_id set not null;
alter table public.user_notes
  add constraint user_notes_user_id_project_id_key unique (user_id, project_id);
alter table public.user_notes drop constraint if exists user_notes_user_id_key;

comment on column public.user_notes.project_id is 'Owning project; required after the cloud-project cutover.';
comment on column public.chat_conversations.project_id is 'Owning project; required after the cloud-project cutover.';

create policy project_papers_read_own
  on public.project_papers for select to authenticated
  using (
    user_id = (select auth.uid())
    and exists (
      select 1 from public.projects p
      where p.id = project_papers.project_id and p.user_id = (select auth.uid())
    )
  );

drop policy user_notes_manage_own on public.user_notes;
create policy user_notes_manage_own on public.user_notes for all
  using (
    user_id = (select auth.uid())
    and exists (select 1 from public.projects p where p.id = user_notes.project_id and p.user_id = (select auth.uid()))
  )
  with check (
    user_id = (select auth.uid())
    and exists (select 1 from public.projects p where p.id = user_notes.project_id and p.user_id = (select auth.uid()))
  );

drop policy chat_conversations_select_own on public.chat_conversations;
drop policy chat_conversations_insert_own on public.chat_conversations;
drop policy chat_conversations_update_own on public.chat_conversations;
drop policy chat_conversations_delete_own on public.chat_conversations;
create policy chat_conversations_select_own on public.chat_conversations for select
  using (user_id = (select auth.uid()) and exists (
    select 1 from public.projects p where p.id = chat_conversations.project_id and p.user_id = (select auth.uid())
  ));
create policy chat_conversations_insert_own on public.chat_conversations for insert
  with check (user_id = (select auth.uid()) and exists (
    select 1 from public.projects p where p.id = chat_conversations.project_id and p.user_id = (select auth.uid())
  ));
create policy chat_conversations_update_own on public.chat_conversations for update
  using (user_id = (select auth.uid()) and exists (
    select 1 from public.projects p where p.id = chat_conversations.project_id and p.user_id = (select auth.uid())
  ))
  with check (user_id = (select auth.uid()) and exists (
    select 1 from public.projects p where p.id = chat_conversations.project_id and p.user_id = (select auth.uid())
  ));
create policy chat_conversations_delete_own on public.chat_conversations for delete
  using (user_id = (select auth.uid()) and exists (
    select 1 from public.projects p where p.id = chat_conversations.project_id and p.user_id = (select auth.uid())
  ));


-- Remove every historical overload; only the canonical project-aware signature remains.
drop function if exists public.prepare_chat_turn(uuid, text, uuid, timestamptz, text, jsonb, boolean);
drop function if exists public.prepare_chat_turn(uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid);
drop function if exists public.prepare_chat_turn(uuid, uuid, text, uuid, timestamptz, text, jsonb, boolean);
drop function if exists public.prepare_chat_turn(uuid, uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid);

create function public.prepare_chat_turn(
  p_conversation_id uuid,
  p_project_id uuid,
  p_title text,
  p_message_id uuid,
  p_message_created_at timestamptz,
  p_text text,
  p_context jsonb default '{}'::jsonb,
  p_include_history boolean default true,
  p_client_request_id uuid default null
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  current_user_id uuid := auth.uid();
  owner_id uuid;
  conversation_project_id uuid;
  history_rows jsonb := '[]'::jsonb;
  existing_message public.chat_messages%rowtype;
  existing_by_request public.chat_messages%rowtype;
  expected_content jsonb := jsonb_build_array(jsonb_build_object('type', 'text', 'text', p_text));
  expected_context jsonb := coalesce(p_context, '{}'::jsonb);
  is_duplicate boolean := false;
begin
  if current_user_id is null then
    raise insufficient_privilege using message = 'authentication required';
  end if;
  if not exists (
    select 1 from public.projects p
    where p.id = p_project_id and p.user_id = current_user_id
  ) then
    raise insufficient_privilege using message = 'project does not belong to this user';
  end if;

  select c.user_id, c.project_id into owner_id, conversation_project_id
  from public.chat_conversations c where c.id = p_conversation_id for update;
  if not found then
    insert into public.chat_conversations (id, user_id, project_id, title)
    values (p_conversation_id, current_user_id, p_project_id, coalesce(nullif(p_title, ''), 'New chat'))
    on conflict (id) do nothing;
    select c.user_id, c.project_id into owner_id, conversation_project_id
    from public.chat_conversations c where c.id = p_conversation_id for update;
  end if;
  if owner_id is distinct from current_user_id then
    raise insufficient_privilege using message = 'conversation does not belong to this user';
  end if;
  if conversation_project_id is distinct from p_project_id then
    raise insufficient_privilege using message = 'conversation belongs to a different project';
  end if;

  select * into existing_message from public.chat_messages where id = p_message_id;
  if existing_message.id is not null then
    if existing_message.conversation_id is distinct from p_conversation_id
      or existing_message.role is distinct from 'user'
      or existing_message.content is distinct from expected_content
      or coalesce(existing_message.context, '{}'::jsonb) is distinct from expected_context
      or existing_message.client_request_id is distinct from p_client_request_id then
      raise unique_violation using message = 'chat message id conflicts with existing message';
    end if;
    is_duplicate := true;
  end if;
  if p_client_request_id is not null then
    select * into existing_by_request from public.chat_messages where client_request_id = p_client_request_id;
    if existing_by_request.id is not null then
      if existing_by_request.id is distinct from p_message_id
        or existing_by_request.conversation_id is distinct from p_conversation_id
        or existing_by_request.role is distinct from 'user'
        or existing_by_request.content is distinct from expected_content
        or coalesce(existing_by_request.context, '{}'::jsonb) is distinct from expected_context then
        raise unique_violation using message = 'client request id conflicts with existing message';
      end if;
      is_duplicate := true;
    end if;
  end if;
  if p_include_history then
    select coalesce(jsonb_agg(to_jsonb(message) order by message.created_at, message.role desc, message.id), '[]'::jsonb)
    into history_rows from (
      select id, role, content, context, created_at from public.chat_messages
      where conversation_id = p_conversation_id and status = 'completed' and id <> p_message_id
    ) message;
  end if;
  if not is_duplicate then
    insert into public.chat_messages (id, conversation_id, role, content, context, status, created_at, client_request_id)
    values (p_message_id, p_conversation_id, 'user', expected_content, expected_context,
      'completed', coalesce(p_message_created_at, now()), p_client_request_id);
  end if;
  return jsonb_build_object(
    'user_id', current_user_id::text, 'project_id', conversation_project_id::text,
    'history', history_rows, 'duplicate', is_duplicate
  );
end;
$$;

revoke all on function public.prepare_chat_turn(uuid, uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid)
  from public, anon, authenticated;
grant execute on function public.prepare_chat_turn(uuid, uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid)
  to authenticated;

commit;
