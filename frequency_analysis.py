from collections import Counter

def flatten_phrase_list(phrases):
    """
    Accepts list of phrases and returns cleaned tokens (split phrases into tokens)
    Useful to count keywords rather than long phrases.
    """
    tokens = []
    for p in phrases:
        parts = p.split()
        
        tokens.append(" ".join(parts))
    return tokens

def compute_topic_frequency(phrases, top_n=10):
    """
    phrases: list of extracted phrases (strings)
    returns: list of tuples (topic, count)
    """
    tokens = flatten_phrase_list(phrases)
    c = Counter(tokens)
    total = sum(c.values()) if c else 1
    ranked = c.most_common(top_n)
    
    ranked_with_weight = [(topic, count, round((count/total)*100,2)) for topic, count in ranked]
    return ranked_with_weight

