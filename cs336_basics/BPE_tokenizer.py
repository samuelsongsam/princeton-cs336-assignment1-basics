import os
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be positive")
    if not isinstance(split_special_token, bytes) or not split_special_token:
        raise ValueError("split_special_token must be nonempty bytes")

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [0]
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    for bi in range(1, desired_num_chunks):
        position = bi * chunk_size
        file.seek(position)  # Start at boundary guess
        tail = b""
        # Read ahead by 4k bytes at a time
        while True:
            block = file.read(4096)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if not block:
                chunk_boundaries.append(file_size)
                break
            data = tail + block
            # Find the special token in the mini chunk
            found_at = data.find(split_special_token)
            if found_at != -1:
                chunk_boundaries.append(position - len(tail) + found_at)
                break
            position += len(block)
            tail = data[-(len(split_special_token) - 1):] if len(split_special_token) > 1 else b""
    chunk_boundaries.append(file_size)
    file.seek(0)
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def parallel_pretokenize(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    patterns = "|".join(re.escape(s) for s in ordered_tokens)
    parts = re.split(patterns, chunk) if ordered_tokens else [chunk]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = Counter()
    for part in parts:
        for match in re.finditer(PAT, part):
            counts[match.group().encode("utf-8")] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not token for token in special_tokens):
        raise ValueError("Special token strings must not be empty")
    #initialize the vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    if vocab_size < len(vocab):
        raise ValueError("vocab_size cannot be smaller than the initial vocabulary")
    merges = []
    if vocab_size == len(vocab):
        return vocab, merges

    #pretokenization
    file_size = os.path.getsize(input_path)
    num_processes = min(4, os.cpu_count() or 1)
    if file_size >= 1024 * 1024 and "<|endoftext|>" in special_tokens and num_processes > 1:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    else:
        boundaries = [0, file_size]
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    if len(tasks) > 1:
        with Pool(processes=min(num_processes, len(tasks))) as pool:
            counts_list = pool.starmap(parallel_pretokenize, tasks)
    else:
        counts_list = [parallel_pretokenize(*task) for task in tasks]

    counts = Counter()
    for chunk_counts in counts_list:
        counts.update(chunk_counts)
    sequences = {}
    frequencies = {}
    pair_counts = Counter()
    pairTOSenquences = {}
    for seq_ID, (pretoken, frequency) in enumerate(counts.items()):
        tokens = tuple(pretoken[i:i + 1] for i in range(len(pretoken)))
        sequences[seq_ID] = tokens
        frequencies[seq_ID] = frequency
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += frequency
            pairTOSenquences.setdefault(pair, set()).add(seq_ID)

    #merge
    while len(vocab) < vocab_size and pair_counts:
        max_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[max_pair] <= 0:
            break
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        for seq_ID in pairTOSenquences[max_pair].copy():
            tokens = sequences[seq_ID]
            frequency = frequencies[seq_ID]
            old_counts = Counter(zip(tokens, tokens[1:]))
            # 从左到右合并，两个旧 token 不会重复参与合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == max_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens = tuple(new_tokens)
            new_counts = Counter(zip(new_tokens, new_tokens[1:]))
            # 更新全局频次，保留负差量
            for pair in old_counts.keys() | new_counts.keys():
                change = (new_counts[pair] - old_counts[pair]) * frequency
                pair_counts[pair] += change
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            # 更新 pair 与序列编号的关联
            for pair in old_counts.keys() - new_counts.keys():
                pairTOSenquences[pair].discard(seq_ID)
                if not pairTOSenquences[pair]:
                    del pairTOSenquences[pair]
            for pair in new_counts.keys() - old_counts.keys():
                pairTOSenquences.setdefault(pair, set()).add(seq_ID)
            sequences[seq_ID] = new_tokens
    return vocab, merges

def main():
    """Train and save byte-preserving JSON artifacts for the handout experiments."""
    import argparse
    import cProfile
    import json
    import threading
    import time
    from pathlib import Path

    import psutil

    parser = argparse.ArgumentParser(description="Train a byte-level BPE vocabulary")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=None,
                        help="Repeat for multiple tokens; default: <|endoftext|>")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--profile", type=Path, help="Save cProfile statistics")
    args = parser.parse_args()
    if args.no_special_tokens and args.special_token:
        parser.error("--no-special-tokens conflicts with --special-token")
    if not args.input_path.is_file():
        parser.error(f"Input file does not exist: {args.input_path}")
    special_tokens = [] if args.no_special_tokens else (
        args.special_token if args.special_token is not None else ["<|endoftext|>"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training: {args.input_path}; target vocabulary: {args.vocab_size}", flush=True)
    process = psutil.Process()
    stop = threading.Event()
    peak_rss = [0]

    def sample_memory():
        while True:
            rss = 0
            for item in [process, *process.children(recursive=True)]:
                try:
                    rss += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak_rss[0] = max(peak_rss[0], rss)
            if stop.wait(0.1):
                return

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    try:
        if profiler:
            profiler.enable()
        vocab, merges = train_bpe(str(args.input_path), args.vocab_size, special_tokens)
    finally:
        elapsed = time.perf_counter() - started
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.profile))
        stop.set()
        monitor.join()

    # Hex preserves arbitrary byte fragments, including incomplete UTF-8 characters.
    (args.output_dir / "vocab.json").write_text(
        json.dumps({str(key): value.hex() for key, value in vocab.items()}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "merges.json").write_text(
        json.dumps([[left.hex(), right.hex()] for left, right in merges], indent=2),
        encoding="utf-8",
    )
    longest = max(vocab.values(), key=len, default=b"")
    metrics = {
        "input_path": str(args.input_path.resolve()),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "training_seconds": elapsed,
        "sampled_peak_sum_rss_mib": peak_rss[0] / (1024 ** 2),
        "memory_note": "Parent plus children RSS sampled every 0.1s; shared pages may be counted multiple times.",
        "longest_token_hex": longest.hex(),
        "longest_token_bytes": len(longest),
        "longest_token_display": longest.decode("utf-8", errors="replace"),
        "artifact_format": "vocab.json: ID -> hex bytes; merges.json: ordered pairs of hex bytes. Decode using bytes.fromhex.",
        "profile_note": "cProfile records the parent process; worker pre-tokenization is not profiled internally.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
