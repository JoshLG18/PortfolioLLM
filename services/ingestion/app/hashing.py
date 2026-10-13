def compute_corpus_hash() -> str:
    raise NotImplementedError(
        "TODO: sha256 over (local doc paths+sha256) + (github readme blob shas) + "
        "(portfolio header/bio content hash) + CHUNKING_VERSION"
    )
