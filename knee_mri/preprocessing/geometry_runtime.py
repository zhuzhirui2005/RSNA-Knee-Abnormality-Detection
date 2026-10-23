# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
"""Single-producer, lazy ZIP lifetime for repeated geometry batches.

The archive index is large; keeping one reader avoids rebuilding its central
directory for every eight-study batch. Cached geometry never needs this reader.
Call close() when the preprocessing stream exits. Do not share across threads.
"""
import json
from pathlib import Path

class GeometryRuntime:

    def __init__(self):
        self.archive = None
        self.reader = None
        self.identity = None
        self.archive_opens = 0
        self.last_build = None

    def source_metadata(self, root, config, source_directory=False):
        root = Path(root)
        if source_directory:
            return {'etag': config['archive_etag']}
        return json.loads((root / 'download/archive_metadata.json').read_text(encoding='utf-8'))

    def open_archive(self, root, metadata, opener):
        identity = (str(Path(root).resolve()), *(metadata.get(k) for k in ('etag', 'generation', 'bytes', 'chunk_size')))
        if self.archive is not None and self.identity != identity:
            self.close()
        if self.archive is None:
            archive, reader, actual = opener(root)
            if any((actual.get(k) != metadata.get(k) for k in ('etag', 'generation', 'bytes', 'chunk_size'))):
                archive.close()
                reader.close()
                raise ValueError('archive changed while opening geometry source')
            self.archive, self.reader, self.identity = (archive, reader, identity)
            self.archive_opens += 1
        return self.archive

    def close(self):
        if self.archive is not None:
            self.archive.close()
        if self.reader is not None:
            self.reader.close()
        self.archive = self.reader = self.identity = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def reusable_geometry(record, uid, series, archive_etag, allowed_builders):
    """Known-equivalent producers retain their original provenance and bytes."""
    if record.get('source_builder_sha256') not in allowed_builders or record.get('source_archive_etag') != archive_etag or record.get('StudyInstanceUID') != uid or (record.get('SeriesInstanceUID') != series['SeriesInstanceUID']):
        return False
    affines = record.get('affines_col_row_origin_lps_mm', [])
    return len(affines) == len(series['files']) and bool(affines) and ('raw_explicit_sides' in record) and ('quality' in record) and all((len(matrix) == 3 and all((len(row) == 3 for row in matrix)) for matrix in affines))
