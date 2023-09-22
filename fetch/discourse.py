from invoke import task

from prompts.chatgpt import ChatGPTPrompt
from load.iterators import DiscourseRecordIterator


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
