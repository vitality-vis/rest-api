-- Let participants edit the text/type of their own feedback, or delete it entirely.
-- Attachments are not editable; deleting an entry also lets the owner remove its images.
begin;

alter table public.feedback_entries add column edited_at timestamptz;

create function public.update_feedback_entry(entry_id uuid, entry_body text, entry_templates text[])
returns timestamptz
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  template_id text;
  current_attachments jsonb;
  edited timestamptz := now();
begin
  if participant is null then raise exception 'Authentication required'; end if;
  if entry_body is null or char_length(entry_body) > 45000
     or entry_templates is null or cardinality(entry_templates) > 1 then
    raise exception 'Invalid entry';
  end if;
  foreach template_id in array entry_templates loop
    if template_id is null or template_id !~ '^[a-z][a-z0-9_-]{0,63}$' then
      raise exception 'Invalid entry';
    end if;
  end loop;
  select attachments into current_attachments
  from public.feedback_entries
  where id = entry_id and user_id = participant
  for update;
  if not found then raise exception 'Feedback entry not found'; end if;
  if btrim(entry_body) = '' and jsonb_array_length(current_attachments) = 0 then
    raise exception 'Entry is empty';
  end if;
  update public.feedback_entries
  set body = entry_body,
      templates = entry_templates,
      content = case
        when content is null then null
        else jsonb_set(content, '{free_text}', to_jsonb(entry_body)) || jsonb_build_object('sections', '[]'::jsonb)
      end,
      edited_at = edited
  where id = entry_id and user_id = participant;
  return edited;
end;
$$;

create function public.delete_feedback_entry(entry_id uuid)
returns jsonb
language plpgsql security definer set search_path = ''
as $$
declare
  participant uuid := auth.uid();
  removed_attachments jsonb;
begin
  if participant is null then raise exception 'Authentication required'; end if;
  delete from public.feedback_entries
  where id = entry_id and user_id = participant
  returning attachments into removed_attachments;
  if not found then raise exception 'Feedback entry not found'; end if;
  return removed_attachments;
end;
$$;

revoke all on function public.update_feedback_entry(uuid, text, text[]) from public, anon;
grant execute on function public.update_feedback_entry(uuid, text, text[]) to authenticated;
revoke all on function public.delete_feedback_entry(uuid) from public, anon;
grant execute on function public.delete_feedback_entry(uuid) to authenticated;

-- Images of a deleted entry (no row left) can be removed by their owner.
create policy feedback_images_delete_orphaned on storage.objects for delete to authenticated
  using (
    bucket_id = 'study-feedback'
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and not exists (select 1 from public.feedback_entries e where e.id::text = split_part(name, '/', 2))
  );

commit;
