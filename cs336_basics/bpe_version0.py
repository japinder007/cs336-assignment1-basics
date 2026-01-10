from collections import Counter
from dataclasses import dataclass
from collections import defaultdict
import os
from typing import BinaryIO
import regex as re
from logging import Logger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)



@dataclass
class Token:
    tokens: tuple[bytes]

@dataclass
class PreToken:
    raw: str
    tokens: list[Token]
    count: int

def process_pre_tokens(pre_tokens: Counter) -> tuple[Counter, dict[tuple[Token], set(PreToken)]]:
    """
    Goes through all the pre-tokens and does the following:
        1. Creates a counter of token pairs.
        2. Creates a reverse index from token pairs to pre-tokens.
    Args:
        pre_tokens: A Counter of pre-tokens.
    Returns:
        A tuple containing the counter of token pairs and the reverse index.
    """
    token_pairs = Counter()
    reverse_index = defaultdict(set)
    for k, v in pre_tokens.items():
        pairs = zip(k.tokens, k.tokens[1:])
        for pair in pairs:
            token_pairs[pair] += v
            reverse_index[pair].add(k)
    return token_pairs, reverse_index


def combine_tokens(token1: Token, token2: Token, token_pair_counts: Counter, reverse_index: dict[tuple[Token], set(PreToken)]):
    """
    Given a token pair, does the following:
        1. Updates the tokens for all pre-tokens containing the pair.
        2. Updates the count for the neighbors of the pair.
        3. Removes the pair from the counter and reverse index.
    Args:
        token1: The first token to combine.
        token2: The second token to combine.
        token_pair_counts: A counter of token pairs.
        reverse_index: A reverse index from token pairs to pre-tokens.
    Returns:
        A tuple containing the counter of token pairs and the reverse index.
    """
    combined_token = Token(token1.tokens + token2.tokens)
    pre_tokens = reverse_index.get((token1, token2), set())
    
    for pre_token in pre_tokens:
        for i in range(len(pre_token.tokens) - 1):
            if (token1, token2) == (pre_token.tokens[i], pre_token.tokens[i + 1]):
                # Replace with the combined token.
                pre_token.tokens[i] = combined_token
                pre_token.tokens[i + 1] = None
            if i - 1 >= 0:
                # Update count for [i - 1, i]
                token_pair_counts[(pre_token.tokens[i - 1], pre_token.tokens[i])] += pre_token.count      
                reverse_index[(pre_token.tokens[i - 1], pre_token.tokens[i])].add(pre_token)
            if i + 2 < len(pre_token.tokens):
                token_pair_counts[(pre_token.tokens[i], pre_token.tokens[i + 2])] += pre_token.count      
                reverse_index[(pre_token.tokens[i], pre_token.tokens[i + 2])].add(pre_token)
        
        # Remove none tokens.
        pre_tokens.tokens = [t for t in pre_token.tokens if t is not None]
    
    # Remove the combined token from the reverse index and token pairs.     
    del reverse_index[(token1, token2)]
    del token_pair_counts[(token1, token2)]
    
    return token_pair_counts, reverse_index

    
def pretokenize_document(document: str) -> Counter: 
    """
    Given a document, return a counter of the pre-tokens in the document.
    
    Args:
        document (string): A document.
    Returns (Counter):
        A counter of the pre-tokens in the document.
    """
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = Counter(re.finditer(PAT, document))
    return tokens


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


def get_pretoken_corpus_counts(documents: list[str]) -> Counter:
    """
    Given a list of documents, return a counter of the pre-tokens in the documents.
    Args:
        documents: A list of documents.
    Returns:
        A counter of the pre-tokens in the documents.
    """
    counter = Counter()
    for document in documents:
        doc_counter = pretokenize_document(document)
        counter += doc_counter
    return counter

def get_max_count_token_pair(token_pair_counts: Counter) -> tuple[Token, Token]:
    """
    Given a counter of token pairs, returns the token pair with the maximum count.
    """
    max_count = token_pair_counts.most_common(1)[0][1]
    # Tie break by lexicographic order.
    max_count_pairs = [pair for pair, count in token_pair_counts.items() if count == max_count]
    return max(max_count_pairs, key=lambda x: (x[0], x[1]))

def get_documents(corpus_filename: str, num_processes: int, split_special_token: bytes) -> list[str]:
    """
    Given a corpus filename, split the corpus into documents based on the split special token.
    
    Args:
        corpus_filename: The filename of the corpus.
        num_processes: The number of processes to use.
        split_special_token: The special token to use to split the corpus into documents.
    Returns:
        A list of documents.
    """
    with open(file_name, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        documents = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            documents.append(chunk)
        return documents
    
def bpe_version_0(documents: list[str], end_of_text: str, num_merges: int):
    vocabulary = [end_of_text] + bytes(range(256))
    
    # Corpus is split based on special tokens.
    pretoken_counts = get_pretoken_corpus_counts(corpus_filename, num_merges, end_of_text)
    log.info(f"Pretoken counts: {pretoken_counts}")
    token_pair_counts, reverse_index = process_pre_tokens(pretoken_counts)
    log.info(f"Token pair counts: {token_pair_counts}")
    log.info(f"Reverse index: {reverse_index}")
    
    rules = []
    for i in range(num_merges):
        max_count_token_pair = get_max_count_token_pair(token_pair_counts)
        log.info(f"Max count token pair {i}: {max_count_token_pair}")
        rules.append(max_count_token_pair)
        log.info(f"Combining tokens {i}: {max_count_token_pair[0]} and {max_count_token_pair[1]}")
        token_pair_counts, reverse_index = combine_tokens(max_count_token_pair[0], max_count_token_pair[1], token_pair_counts, reverse_index)
        log.info(f"Token pair counts {i}: {token_pair_counts}")
        log.info(f"Reverse index {i}: {reverse_index}")
        vocabulary.append(max_count_token_pair[0].tokens + max_count_token_pair[1].tokens)
        log.info(f"Vocabulary {i}: {vocabulary}")
    
    return rules, vocabulary


if __name__ == "__main__":
    bpe_corpus = """low low low low low
    lower lower widest widest
    newest newest newest newest newest newest"""

    end_of_text = "<|endoftext|>"
    num_merges = 6
    rules, vocabulary = bpe_version_0(bpe_corpus, end_of_text, num_merges)
    print(rules)
    print(vocabulary)
