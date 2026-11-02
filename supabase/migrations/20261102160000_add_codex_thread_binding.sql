-- Persist the model-side Codex thread bound to each product conversation.
-- Chat messages remain the product record and recovery source; these columns
-- only identify the resumable Codex Harness thread.
alter table public.chat_conversations
  add column codex_thread_id text,
  add column codex_thread_schema_version text,
  add column codex_thread_bound_at timestamptz,
  add constraint chat_conversations_codex_thread_binding_complete check (
    (codex_thread_id is null
      and codex_thread_schema_version is null
      and codex_thread_bound_at is null)
    or
    (nullif(btrim(codex_thread_id), '') is not null
      and nullif(btrim(codex_thread_schema_version), '') is not null
      and codex_thread_bound_at is not null)
  );

create unique index chat_conversations_codex_thread_id_unique
  on public.chat_conversations (codex_thread_id)
  where codex_thread_id is not null;

comment on column public.chat_conversations.codex_thread_id is
  'Opaque persisted Codex Harness thread ID used to resume model context.';
comment on column public.chat_conversations.codex_thread_schema_version is
  'Application instruction/tool schema version required to resume the thread.';
comment on column public.chat_conversations.codex_thread_bound_at is
  'Time the current Codex thread binding was committed.';

-- Conversation rows are visible to their owners, but thread bindings are a
-- trusted server concern. Reject direct authenticated-client attempts to set,
-- replace, or clear them; server requests use the service_role JWT.
create or replace function public.protect_chat_conversation_codex_binding()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
  if coalesce(auth.role(), '') <> 'service_role' then
    if tg_op = 'INSERT' then
      if new.codex_thread_id is not null
        or new.codex_thread_schema_version is not null
        or new.codex_thread_bound_at is not null
      then
        raise exception 'Codex thread binding is server-managed'
          using errcode = '42501';
      end if;
    elsif new.codex_thread_id is distinct from old.codex_thread_id
      or new.codex_thread_schema_version is distinct from old.codex_thread_schema_version
      or new.codex_thread_bound_at is distinct from old.codex_thread_bound_at
    then
      raise exception 'Codex thread binding is server-managed'
        using errcode = '42501';
    end if;
  end if;
  return new;
end;
$$;

create trigger chat_conversations_protect_codex_binding
before insert or update on public.chat_conversations
for each row
execute function public.protect_chat_conversation_codex_binding();
