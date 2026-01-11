from collections import Counter
from dataclasses import dataclass
from collections import defaultdict
import os
from typing import Any, BinaryIO, Tuple, Set, List, DefaultDict
import regex as re
from logging import Logger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Token:
    value: bytes

@dataclass
class PreToken:
    raw: str
    tokens: List[Token]
    count: int

TokenPair = Tuple[Token, Token]
PairCounts = Counter[TokenPair]
ReverseIndex = DefaultDict[TokenPair, Set[int]]

def process_pre_tokens(pre_tokens: List[PreToken]) -> tuple[PairCounts, ReverseIndex]:
    """
    Goes through all the pre-tokens and does the following:
        1. Creates a counter of token pairs.
        2. Creates a reverse index from token pairs to pre-tokens.
    Args:
        pre_tokens: A list of pre-tokens.
    Returns:
        A tuple containing the counter of token pairs and the reverse index.
    """
    token_pairs : PairCounts = Counter()
    reverse_index : ReverseIndex = defaultdict(set)
    
    for i, pre_token in enumerate(pre_tokens):
        for pair in zip(pre_token.tokens, pre_token.tokens[1:]):
            token_pairs[pair] += pre_token.count
            reverse_index[pair].add(i)
    return token_pairs, reverse_index


def get_merged_tokens(
    pre_tokens: List[PreToken], 
    combined_token: TokenPair, 
    reverse_index: ReverseIndex) -> List[Tuple[int, List[Token]]]:
    """
    Given a token pair to combine, identfies all the pre-tokens containing the pair, and returns a list of tuples of the index of the pre-token and the new tokens.
    Args:
        pre_tokens: A list of pre-tokens.
        combined_token: A token pair to combine.
        reverse_index: A reverse index from token pairs to pre-tokens.
    Returns:
        A list of tuples of the index of the pre-token and the new tokens.
    """
    indices = reverse_index.get(combined_token)
    if not indices:
        # The combined token is not in the reverse index.
        return []
    
    result = []
    new_combined_token = Token(value=combined_token[0].value + combined_token[1].value)
    for i in indices:
        pre_token = pre_tokens[i]
        old_tokens = pre_token.tokens
        new_tokens = []
        j = 0
        while j < (len(old_tokens) - 1):
            if combined_token == (old_tokens[j], old_tokens[j + 1]):
                new_tokens.append(new_combined_token)
                j += 2
            else:
                new_tokens.append(old_tokens[j])
                j += 1
        if j == len(old_tokens) - 1:
            new_tokens.append(old_tokens[j])
        
        result.append((i, new_tokens))
    return result
    
def combine_tokens(
    pre_tokens: List[PreToken],
    combined_token: Tuple[Token, Token], 
    token_pair_counts: Counter[Tuple[Token, Token]], 
    reverse_index: ReverseIndex
) -> Tuple[Counter[Tuple[Token, Token]], ReverseIndex]:
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
    merged_tokens = get_merged_tokens(pre_tokens, combined_token, reverse_index)
    for i, new_tokens in merged_tokens:
        pre_token = pre_tokens[i]
        old_pairs = zip(pre_token.tokens, pre_token.tokens[1:])
        for old_pair in old_pairs:
            # Delete the old token pairs from the counter.
            token_pair_counts[old_pair] -= pre_token.count
            if token_pair_counts[old_pair] <= 0:
                token_pair_counts.pop(old_pair, None)
            s = reverse_index.get(old_pair)
            if s is not None:
                s.discard(i)
                if not s:
                    reverse_index.pop(old_pair, None)
        
        pre_token.tokens = new_tokens
        for new_pair in zip(new_tokens, new_tokens[1:]):
            token_pair_counts[new_pair] += pre_token.count
            reverse_index[new_pair].add(i)
        
    return token_pair_counts, reverse_index
    
def pretokenize_document(document: str) -> Counter[str]: 
    """
    Given a document, return a counter of the pre-tokens in the document.
    
    Args:
        document (string): A document.
    Returns (Counter[PreToken]):
        A counter of the pre-tokens in the document.
    """
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = Counter(m.group(0) for m in re.finditer(PAT, document))
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


def get_pretoken_corpus_counts(documents: list[str]) -> List[PreToken]:
    """
    Given a list of documents, return a counter of the pre-tokens in the documents.
    Args:
        documents: A list of documents.
    Returns:
        A counter of the pre-tokens in the documents.
    """
    counter: Counter[str] = Counter()
    for document in documents:
        counter += pretokenize_document(document)
    
    pretokens: List[PreToken] = []
    for k, v in counter.items():
        b = k.encode("utf-8")
        pretoken_tokens = [Token(value=bytes([x])) for x in b]
        pretokens.append(PreToken(raw=k, tokens=pretoken_tokens, count=v))
    return pretokens

def get_max_count_token_pair(token_pair_counts: Counter[Tuple[Token, Token]]) -> Tuple[Token, Token]:
    """
    Given a counter of token pairs, returns the token pair with the maximum count.
    """
    if not token_pair_counts:
        raise ValueError("Token pair counts is empty")
    
    return max(token_pair_counts.items(), key=lambda x: (x[1], x[0][0].value, x[0][1].value))[0]

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
    with open(corpus_filename, "rb") as f:
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
    
def bpe_version_0(documents: list[str], end_of_text: bytes, num_merges: int):
    vocabulary = [end_of_text] + [bytes([i]) for i in range(256)]
    
    # Corpus is split based on special tokens.
    pretoken_counts: List[PreToken] = get_pretoken_corpus_counts(documents)
    log.info(f"Pretoken counts: {pretoken_counts}")
    token_pair_counts, reverse_index = process_pre_tokens(pretoken_counts)
    log.info(f"Token pair counts: {token_pair_counts}")
    log.info(f"Reverse index: {reverse_index}")
    
    rules = []
    for i in range(num_merges):
        max_count_token_pair = get_max_count_token_pair(token_pair_counts)
        log.info(f"Max count token pair {i}: {max_count_token_pair}")
        rules.append(max_count_token_pair)
        token_pair_counts, reverse_index = combine_tokens(max_count_token_pair, token_pair_counts, reverse_index)
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
