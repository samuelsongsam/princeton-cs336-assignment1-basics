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
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def parallel_pretokenize(
        input_path: str,
        start: int,
        end: int,
        special_tokens: list[str],
):
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        patterns = '|'.join(re.escape(s) for s in special_tokens)
        parts = re.split(patterns, chunk)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counts = Counter()
        for part in parts:
            pre_tokens = re.findall(PAT, part)
            for pre_token in pre_tokens:
                counts.update([pre_token.encode("utf-8")])
    return counts

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens:  list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :
    if vocab_size <= 256:
        raise ValueError("Invalid vocabulary size!")
    if not special_tokens:
        raise ValueError("special_tokens must not be empty")

    with open(input_path, 'rb') as f:

        #initialize the vocabulary
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])
        for i in range(len(special_tokens)):
            vocab[256 + i] = special_tokens[i].encode('utf-8')

        #pretokenization
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=num_processes) as pool:
            counts_list = pool.starmap(parallel_pretokenize, tasks)

        token_counts = Counter()
        for chunk_counts in counts_list:
            for pretoken, frequency in chunk_counts.items():
                tokens = tuple(
                    pretoken[i:i + 1] for i in range(len(pretoken))
                )
                token_counts[tokens] += frequency

        pair_counts = Counter()
        for tokens, frequency in token_counts.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += frequency
