-- Prepare an authenticated chat turn in one transaction and one HTTP request.
-- Supabase validates the caller JWT; auth.uid() is the authoritative owner.
create or replace function public.prepare_chat_turn(
  p_conversation_id uuid,
  p_title text,
  p_message_id uuid,
  p_message_created_at timestamptz,
  p_text text,
  p_context jsonb default '{}'::jsonb,
  p_include_history boolean default true
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

  if p_include_history then
    select coalesce(jsonb_agg(to_jsonb(m) order by m.created_at, m.role desc, m.id), '[]'::jsonb)
      into history_rows
    from (
      select id, role, content, context, created_at
      from public.chat_messages
      where conversation_id = p_conversation_id
        and status = 'completed'
        and (p_message_id is null or id <> p_message_id)
    ) as m;
  end if;

  insert into public.chat_messages (
    id, conversation_id, role, content, context, status, created_at
  )
  values (
    coalesce(p_message_id, gen_random_uuid()),
    p_conversation_id,
    'user',
    jsonb_build_array(jsonb_build_object('type', 'text', 'text', p_text)),
    coalesce(p_context, '{}'::jsonb),
    'completed',
    coalesce(p_message_created_at, now())
  )
  on conflict (id) do nothing;

  return jsonb_build_object(
    'user_id', current_user_id::text,
    'history', history_rows
  );
end;
$$;

revoke all on function public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean
) from public, anon, authenticated;
grant execute on function public.prepare_chat_turn(
  uuid, text, uuid, timestamptz, text, jsonb, boolean
) to authenticated;
