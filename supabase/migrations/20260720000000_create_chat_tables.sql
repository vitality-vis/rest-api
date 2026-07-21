-- Logged-in users' chat sessions. Guest chats remain in browser localStorage.
create table public.chat_conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null default 'New chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index chat_conversations_user_updated_idx
  on public.chat_conversations (user_id, updated_at desc);

-- User-visible messages in a conversation. Assistant response metadata is optional.
create table public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null
    references public.chat_conversations (id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content jsonb not null default '[]'::jsonb,
  status text not null default 'completed'
    check (status in ('streaming', 'completed', 'failed')),
  duration_ms integer,
  model text,
  input_tokens integer,
  output_tokens integer,
  error_message text,
  created_at timestamptz not null default now()
);

create index chat_messages_conversation_created_idx
  on public.chat_messages (conversation_id, created_at asc, id asc);

-- Every message insertion makes its parent conversation most recently updated.
create or replace function public.touch_conversation_updated_at()
returns trigger
language plpgsql
as $$
begin
  update public.chat_conversations
  set updated_at = now()
  where id = new.conversation_id;
  return new;
end;
$$;

create trigger chat_messages_touch_conversation
after insert on public.chat_messages
for each row
execute function public.touch_conversation_updated_at();

-- New tables are not automatically exposed to API roles. Only signed-in users
-- receive the minimum SQL privileges; RLS below restricts rows to their owner.
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.chat_conversations to authenticated;
grant select, insert, delete on public.chat_messages to authenticated;

alter table public.chat_conversations enable row level security;
alter table public.chat_messages enable row level security;

-- Conversations are visible and mutable only to their owning user.
create policy chat_conversations_select_own
  on public.chat_conversations for select
  using (auth.uid() = user_id);

create policy chat_conversations_insert_own
  on public.chat_conversations for insert
  with check (auth.uid() = user_id);

create policy chat_conversations_update_own
  on public.chat_conversations for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy chat_conversations_delete_own
  on public.chat_conversations for delete
  using (auth.uid() = user_id);

-- Messages inherit access from their owning conversation.
create policy chat_messages_select_own
  on public.chat_messages for select
  using (
    exists (
      select 1 from public.chat_conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

create policy chat_messages_insert_own
  on public.chat_messages for insert
  with check (
    exists (
      select 1 from public.chat_conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );

create policy chat_messages_delete_own
  on public.chat_messages for delete
  using (
    exists (
      select 1 from public.chat_conversations c
      where c.id = conversation_id and c.user_id = auth.uid()
    )
  );
