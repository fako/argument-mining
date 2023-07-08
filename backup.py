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
