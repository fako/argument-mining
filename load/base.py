from json import loads
from tqdm import tqdm


def load_dump_file(dump_file_path, formatter, limit=None):
    with open(dump_file_path) as dump_file:
        batch_count = 0
        for _ in dump_file.readlines():
            batch_count += 1
        dump_file.seek(0)
        for line_number, line in tqdm(enumerate(dump_file), total=batch_count):
            if limit and line_number >= limit:
                return
            yield [formatter(obj) for obj in loads(line)]
