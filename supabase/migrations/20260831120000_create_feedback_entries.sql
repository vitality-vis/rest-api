-- Simple feedback: one immutable submitted entry with an attachment manifest.
-- Images are uploaded first; the RPC publishes the entry only after validation.
create table public.feedback_entries (
  id uuid primary key,
  user_id uuid not null references auth.users(id),
  body text not null check (char_length(body) <= 50000),
  templates text[] not null default '{}',
  template_version integer not null default 1,
  attachments jsonb not null default '[]'::jsonb,
  submitted_at timestamptz not null default now()
);
create index feedback_entries_user_date on public.feedback_entries(user_id, submitted_at desc);
alter table public.feedback_entries enable row level security;
revoke all on public.feedback_entries from anon, authenticated;
grant select on public.feedback_entries to authenticated;
create policy feedback_read_own on public.feedback_entries for select to authenticated
  using (user_id = (select auth.uid()));

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('study-feedback', 'study-feedback', false, 5242880,
        array['image/png', 'image/jpeg', 'image/webp']);

create policy feedback_images_read_own on storage.objects for select to authenticated
  using (bucket_id = 'study-feedback' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy feedback_images_insert_own on storage.objects for insert to authenticated
  with check (
    bucket_id = 'study-feedback'
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and array_length(string_to_array(name, '/'), 1) = 3
    and not exists (select 1 from public.feedback_entries e where e.id::text = split_part(name, '/', 2))
  );
create policy feedback_images_update_pending on storage.objects for update to authenticated
  using (
    bucket_id = 'study-feedback' and (storage.foldername(name))[1] = (select auth.uid())::text
    and not exists (select 1 from public.feedback_entries e where e.id::text = split_part(name, '/', 2))
  ) with check (
    bucket_id = 'study-feedback' and (storage.foldername(name))[1] = (select auth.uid())::text
    and array_length(string_to_array(name, '/'), 1) = 3
    and not exists (select 1 from public.feedback_entries e where e.id::text = split_part(name, '/', 2))
  );

create function public.submit_feedback_entry(
  entry_id uuid, entry_body text, entry_templates text[], entry_attachments jsonb
) returns uuid
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  attachment jsonb;
  object_path text;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if entry_id is null then raise exception 'Entry ID required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(entry_id::text, 0));
  if exists (select 1 from public.feedback_entries where id = entry_id and user_id = participant) then
    return entry_id;
  end if;
  if entry_body is null or char_length(entry_body) > 50000
     or entry_templates is null or cardinality(entry_templates) > 6
     or not (entry_templates <@ array['workflow','ai-features','ai-responses','other-tools','system','other'])
     or entry_attachments is null or jsonb_typeof(entry_attachments) <> 'array' then
    raise exception 'Invalid entry';
  end if;
  if jsonb_array_length(entry_attachments) > 5
     or (btrim(entry_body) = '' and jsonb_array_length(entry_attachments) = 0) then
    raise exception 'Entry is empty or has too many attachments';
  end if;
  for attachment in select value from jsonb_array_elements(entry_attachments) loop
    object_path := attachment->>'path';
    if object_path is null or split_part(object_path, '/', 1) <> participant::text
       or split_part(object_path, '/', 2) <> entry_id::text
       or array_length(string_to_array(object_path, '/'), 1) <> 3
       or not exists (
         select 1 from storage.objects where bucket_id = 'study-feedback' and name = object_path
       ) then raise exception 'Invalid or missing attachment'; end if;
  end loop;
  insert into public.feedback_entries(id, user_id, body, templates, attachments)
  values (entry_id, participant, entry_body, entry_templates, entry_attachments);
  return entry_id;
end;
$$;
revoke all on function public.submit_feedback_entry(uuid, text, text[], jsonb) from public, anon;
grant execute on function public.submit_feedback_entry(uuid, text, text[], jsonb) to authenticated;
