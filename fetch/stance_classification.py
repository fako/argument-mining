from invoke import task

from constants import STANCE_ZERO_SHOT_TARGETS
from prompts.chatgpt import ChatGPTPrompt
from embeddings.chatgpt import ChatGPTEmbeddings
from load.iterators import PostRecordIterator


@task()
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


@task()
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


@task()
def embeddings_stance_classification(ctx, scope, dry_run=False, limit=None):
    limit = int(limit) if limit is not None else None
    chatgpt = ChatGPTEmbeddings(ctx.config, "claims")
    post_iterator = PostRecordIterator(
        output_directory=ctx.config.directories.output,
        scope=scope,
        max_text_length=ChatGPTPrompt.text_length_limit,
        limit=limit,
        prompts={"splitting": ChatGPTPrompt(ctx.config, "splitting", is_list=True)},
    )
    for identifier, record, prompts in post_iterator:
        splitting = prompts.get("splitting", None)
        if not splitting:
            print("Missing splitting prompt:", identifier)
            continue
        for splits in splitting:
            if not isinstance(splits, dict):
                print("Invalid splits type:", type(splits))
                print(splits)
                continue
            for premise in splits.get("premises", []):
                cache_key = chatgpt.get_cache_key(identifier, premise)
                chatgpt.fetch(cache_key, premise, dry_run=dry_run)
            conclusion = splits.get("conclusion", None)
            if conclusion:
                cache_key = chatgpt.get_cache_key(identifier, conclusion)
                chatgpt.fetch(cache_key, conclusion)
