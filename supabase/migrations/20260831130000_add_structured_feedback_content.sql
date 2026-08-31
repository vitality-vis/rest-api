-- Apply AFTER 20260831120000_create_feedback_entries.sql.
-- Preserve legacy entries/clients; new entries have a complete JSONB document.
-- Includes the single-template restriction for new submissions.
begin;

alter table public.feedback_entries add column content jsonb;
update public.feedback_entries
set content = jsonb_build_object(
  'schema_version', 1, 'free_text', body, 'sections', '[]'::jsonb,
  'legacy_templates', to_jsonb(templates), 'attachments', attachments
);

-- Older clients can continue to use the original submission RPC.
create function public.feedback_legacy_content()
returns trigger language plpgsql set search_path = '' as $$
begin
  if new.content is null then
    new.content := jsonb_build_object(
      'schema_version', 1, 'free_text', new.body, 'sections', '[]'::jsonb,
      'legacy_templates', to_jsonb(new.templates), 'attachments', new.attachments
    );
  end if;
  return new;
end;
$$;
create trigger feedback_default_legacy_content
before insert on public.feedback_entries
for each row execute function public.feedback_legacy_content();
alter table public.feedback_entries alter column content set not null;
alter table public.feedback_entries add constraint feedback_content_object
  check (jsonb_typeof(content) = 'object');

create function public.submit_feedback_entry_json(entry_id uuid, entry_content jsonb)
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
    if template_id is null or not (template_id = any(array['workflow','ai-features','ai-responses','other-tools','system','other']))
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

  -- Reuse authentication, path ownership, attachment existence and length checks.
  -- Both insertion and JSON update occur in this same transaction.
  perform public.submit_feedback_entry(entry_id, readable_body, selected_templates, entry_content->'attachments');
  update public.feedback_entries
  set content = entry_content, template_version = 2
  where id = entry_id and user_id = participant;
  return entry_id;
end;
$$;
revoke all on function public.submit_feedback_entry_json(uuid, jsonb) from public, anon;
grant execute on function public.submit_feedback_entry_json(uuid, jsonb) to authenticated;

-- Apply the same rule to legacy submissions; existing records stay unchanged.
create function public.enforce_feedback_single_template()
returns trigger language plpgsql set search_path = '' as $$
begin
  if cardinality(new.templates) > 1
     or jsonb_array_length(new.content->'sections') > 1 then
    raise exception 'Each feedback entry may use at most one template'
      using errcode = '23514';
  end if;
  return new;
end;
$$;

create trigger feedback_single_template
before insert or update on public.feedback_entries
for each row execute function public.enforce_feedback_single_template();

commit;
