-- Tab visibility is an account preference for authenticated users, not a
-- browser-local setting. Guests continue to keep their temporary state local.
alter table public.chat_conversations
  add column is_closed boolean not null default false;
