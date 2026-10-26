-- Add replay protection after the original prepare_chat_turn migration.
-- Safe whether the original migration was applied previously or in this run.
alter table public.chat_messages
  add column if not exists client_request_id uuid;

create unique index if not exists chat_messages_client_request_id_unique
  on public.chat_messages (client_request_id)
  where client_request_id is not null;

-- PostgreSQL overloads functions by argument types, so remove both the old
-- signature and any partial deployment of this replacement before recreating it.
drop function if exists public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean
);
drop function if exists public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid
);

create function public.prepare_chat_turn(
  p_conversation_id uuid,
  p_title text,
  p_message_id uuid,
  p_message_created_at timestamptz,
  p_text text,
  p_context jsonb default '{}'::jsonb,
  p_include_history boolean default true,
  p_client_request_id uuid default null
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  owner_id uuid;
  history_rows jsonb := '[]'::jsonb;
  current_user_id uuid := auth.uid();
  existing_message public.chat_messages%rowtype;
  existing_by_request public.chat_messages%rowtype;
  expected_content jsonb := jsonb_build_array(
    jsonb_build_object('type', 'text', 'text', p_text)
  );
  expected_context jsonb := coalesce(p_context, '{}'::jsonb);
  is_duplicate boolean := false;
begin
  if current_user_id is null then
    raise insufficient_privilege using message = 'authentication required';
  end if;

  select user_id into owner_id
  from public.chat_conversations
  where id = p_conversation_id
  for update;

  if owner_id is null then
    insert into public.chat_conversations (id, user_id, title)
    values (p_conversation_id, current_user_id, coalesce(nullif(p_title, ''), 'New chat'))
    on conflict (id) do nothing;

    select user_id into owner_id
    from public.chat_conversations
    where id = p_conversation_id
    for update;
  end if;

  if owner_id is distinct from current_user_id then
    raise insufficient_privilege using message = 'conversation does not belong to this user';
  end if;

  select * into existing_message
  from public.chat_messages
  where id = p_message_id;

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
    select * into existing_by_request
    from public.chat_messages
    where client_request_id = p_client_request_id;

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
    select coalesce(jsonb_agg(to_jsonb(m) order by m.created_at, m.role desc, m.id), '[]'::jsonb)
      into history_rows
    from (
      select id, role, content, context, created_at
      from public.chat_messages
      where conversation_id = p_conversation_id
        and status = 'completed'
        and id <> p_message_id
    ) as m;
  end if;

  if not is_duplicate then
    insert into public.chat_messages (
      id, conversation_id, role, content, context, status, created_at,
      client_request_id
    )
    values (
      p_message_id,
      p_conversation_id,
      'user',
      expected_content,
      expected_context,
      'completed',
      coalesce(p_message_created_at, now()),
      p_client_request_id
    );
  end if;

  return jsonb_build_object(
    'user_id', current_user_id::text,
    'history', history_rows,
    'duplicate', is_duplicate
  );
end;
$$;

revoke all on function public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid
) from public, anon, authenticated;
grant execute on function public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean, uuid
) to authenticated;
