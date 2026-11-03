-- Read-only post-migration reconciliation queries.
-- Run immediately after the atomic cutover and investigate every unexpected
-- nonzero invariant count. runtime shelf behavior is exclusively membership-based.

select count(*) as user_notes_without_project
from public.user_notes
where project_id is null;

select count(*) as chat_conversations_without_project
from public.chat_conversations
where project_id is null;

select count(*) as note_project_ownership_mismatches
from public.user_notes as notes
join public.projects as projects on projects.id = notes.project_id
where notes.user_id <> projects.user_id;

select count(*) as conversation_project_ownership_mismatches
from public.chat_conversations as conversations
join public.projects as projects on projects.id = conversations.project_id
where conversations.user_id <> projects.user_id;

select count(*) as project_paper_ownership_mismatches
from public.project_papers as memberships
left join public.projects as projects on projects.id = memberships.project_id
left join public.user_papers as papers on papers.id = memberships.user_paper_id
where projects.id is null
   or papers.id is null
   or memberships.user_id <> projects.user_id
   or memberships.user_id <> papers.user_id;

