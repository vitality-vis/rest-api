-- Simplify feedback submissions: free text + attachments.
-- Template ids are optional history metadata; no product allow-list.
begin;

create or replace function public.submit_feedback_entry(
  entry_id uuid, entry_body text, entry_templates text[], entry_attachments jsonb
) returns uuid
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  attachment jsonb;
  object_path text;
  template_id text;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if entry_id is null then raise exception 'Entry ID required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(entry_id::text, 0));
  if exists (select 1 from public.feedback_entries where id = entry_id and user_id = participant) then
    return entry_id;
  end if;
  if entry_body is null or char_length(entry_body) > 50000
     or entry_templates is null or cardinality(entry_templates) > 6
     or entry_attachments is null or jsonb_typeof(entry_attachments) <> 'array' then
    raise exception 'Invalid entry';
  end if;
  foreach template_id in array entry_templates loop
    if template_id is null or template_id !~ '^[a-z][a-z0-9_-]{0,63}$' then
      raise exception 'Invalid entry';
    end if;
  end loop;
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

create or replace function public.submit_feedback_entry_json(entry_id uuid, entry_content jsonb)
returns uuid language plpgsql security definer set search_path = '' as $$
declare
  participant uuid := auth.uid();
  section jsonb;
  question jsonb;
  selected_templates text[] := '{}';
  question_ids text[];
  template_id text;
  question_id text;
  readable_body text;
  answer_characters integer;
  has_answer boolean;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if entry_id is null then raise exception 'Entry ID required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(entry_id::text, 0));
  if exists (select 1 from public.feedback_entries where id = entry_id and user_id = participant) then
    return entry_id;
  end if;
  if jsonb_typeof(entry_content) is distinct from 'object'
     or entry_content->'schema_version' is distinct from '2'::jsonb
     or jsonb_typeof(entry_content->'free_text') is distinct from 'string'
     or jsonb_typeof(entry_content->'sections') is distinct from 'array'
     or jsonb_typeof(entry_content->'attachments') is distinct from 'array'
     or octet_length(entry_content::text) > 500000 then
    raise exception 'Invalid feedback JSON';
  end if;
  if jsonb_array_length(entry_content->'sections') > 1 then
    raise exception 'Each feedback entry may use at most one template';
  end if;
  readable_body := entry_content->>'free_text';
  answer_characters := char_length(readable_body);
  has_answer := btrim(readable_body) <> '';
  for section in select value from jsonb_array_elements(entry_content->'sections') loop
    template_id := section->>'template_id';
    if template_id is null
       or template_id !~ '^[a-z][a-z0-9_-]{0,63}$'
       or template_id = any(selected_templates)
       or jsonb_typeof(section->'template_label') is distinct from 'string'
       or char_length(section->>'template_label') > 300
       or jsonb_typeof(section->'questions') is distinct from 'array' then
      raise exception 'Invalid template section';
    end if;
    if jsonb_array_length(section->'questions') not between 1 and 10 then
      raise exception 'Invalid question count';
    end if;
    selected_templates := array_append(selected_templates, template_id);
    question_ids := '{}';
    readable_body := readable_body || E'\n\n' || (section->>'template_label');
    for question in select value from jsonb_array_elements(section->'questions') loop
      question_id := question->>'id';
      if question_id is null or question_id !~ '^[a-z][a-z0-9_]{0,63}$'
         or question_id = any(question_ids)
         or jsonb_typeof(question->'prompt') is distinct from 'string'
         or char_length(question->>'prompt') > 1000
         or jsonb_typeof(question->'answer') is distinct from 'string' then
        raise exception 'Invalid question or answer';
      end if;
      question_ids := array_append(question_ids, question_id);
      answer_characters := answer_characters + char_length(question->>'answer');
      has_answer := has_answer or btrim(question->>'answer') <> '';
      if btrim(question->>'answer') <> '' then
        readable_body := readable_body || E'\n\n' || (question->>'prompt') || E'\n' || (question->>'answer');
      end if;
    end loop;
  end loop;
  if answer_characters > 45000 or (not has_answer and jsonb_array_length(entry_content->'attachments') = 0) then
    raise exception 'Entry is empty or too long';
  end if;

  perform public.submit_feedback_entry(entry_id, readable_body, selected_templates, entry_content->'attachments');
  update public.feedback_entries
  set content = entry_content, template_version = 2
  where id = entry_id and user_id = participant;
  return entry_id;
end;
$$;

revoke all on function public.submit_feedback_entry(uuid, text, text[], jsonb) from public, anon;
grant execute on function public.submit_feedback_entry(uuid, text, text[], jsonb) to authenticated;
revoke all on function public.submit_feedback_entry_json(uuid, jsonb) from public, anon;
grant execute on function public.submit_feedback_entry_json(uuid, jsonb) to authenticated;

commit;
