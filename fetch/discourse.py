from invoke import task

from prompts.chatgpt import ChatGPTPrompt
from load.iterators import DiscourseRecordIterator
from embeddings.chatgpt import ChatGPTEmbeddings


MODEL = "gpt-4-0613"


@task()
def assess_discourse(ctx, scope, discourse, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTPrompt(ctx.config, "assessment", model=MODEL, context={
        "topic": discourse
    })
    post_iterator = DiscourseRecordIterator(
        project=discourse,
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=chatgpt.text_length_limit - 200,
        limit=limit,
        cut_off_text=True
    )
    for identifier, record, aspects in post_iterator:
        chatgpt.fetch(identifier, record["text"], dry_run=dry_run)


@task()
def embeddings_discourse(ctx, scope, discourse, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTEmbeddings(ctx.config, "claims")
    post_iterator = DiscourseRecordIterator(
        project=discourse,
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=ChatGPTPrompt.text_length_limit - 200,
        limit=limit,
        prompts={"assessment": ChatGPTPrompt(ctx.config, "assessment")},
        cut_off_text=True
    )
    for identifier, record, prompts in post_iterator:
        assessment = prompts.get("assessment", None)
        if not assessment:
            print("Missing assessment prompt:", identifier)
            continue

        for premise in assessment.get("premises", []):
            cache_key = chatgpt.get_cache_key(identifier, premise)
            chatgpt.fetch(cache_key, premise, dry_run=dry_run)
        conclusion = assessment.get("conclusion", None)
        if conclusion:
            cache_key = chatgpt.get_cache_key(identifier, conclusion)
            chatgpt.fetch(cache_key, conclusion)


@task()
def split_discourse(ctx, scope, discourse, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTPrompt(ctx.config, "splitting_stance", model=MODEL, context={
        "topic": discourse
    })
    post_iterator = DiscourseRecordIterator(
        project=discourse,
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=ChatGPTPrompt.text_length_limit - 200,
        limit=limit,
        prompts={"assessment": ChatGPTPrompt(ctx.config, "assessment")},
        cut_off_text=True
    )
    for identifier, record, prompts in post_iterator:
        assessment = prompts.get("assessment", None)
        if not assessment:
            print("Missing assessment prompt:", identifier)
            continue
        is_relevant = assessment.get("is_relevant") == "yes"
        if not is_relevant:
            print("Irrelevant document:", identifier)
            continue
        chatgpt.fetch(identifier, record["text"], dry_run=dry_run)
