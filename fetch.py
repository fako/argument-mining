import os
import json

from invoke import task

from constants import STANCE_ZERO_SHOT_TARGETS
from prompts.chatgpt import ChatGPTPrompt
from load import PostRecordIterator


@task(name="classify")
def classify_stance_classification(ctx, scope, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTPrompt(ctx.config, "classification", context={
        "topics": list(STANCE_ZERO_SHOT_TARGETS.values())
    })
    post_iterator = PostRecordIterator(
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=chatgpt.text_length_limit,
        limit=limit
    )
    for identifier, record, aspects in post_iterator:
        chatgpt.fetch(identifier, record["text"], dry_run=dry_run)


@task(name="split")
def split_stance_classification(ctx, scope, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTPrompt(ctx.config, "splitting")
    post_iterator = PostRecordIterator(
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=chatgpt.text_length_limit,
        limit=limit
    )
    for identifier, record, aspects in post_iterator:
        chatgpt.fetch(identifier, record["text"], dry_run=dry_run)
