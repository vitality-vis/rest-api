-- Non-visible, per-message UI context (for example selected-paper snapshots).
alter table public.chat_messages
  add column context jsonb not null default '{}'::jsonb;
