import os
from invoke import task


@task(name="backup-output")
def backup_stance_classification_output(ctx):
    # Syncing local data to a harddrive
    # -z means use compression
    # -r means recursive
    # -t means preserve creation and modification times
    # -h means human readable output
    # -v means verbose
    ctx.run(f"rsync -zrthv --progress {ctx.config.directories.output} {ctx.config.directories.backup}")


@task(name="sync-output")
def sync_discourse_output(ctx, remote):
    remote_config = getattr(ctx.config.remotes, remote)
    remote_data_directory = os.path.join(remote_config.data, "output/")
    remote_source = f"{remote_config.user}@{remote_config.host}:{remote_data_directory}"
    ctx.run(f"rsync -zrthv --progress {remote_source} {ctx.config.directories.output}")
