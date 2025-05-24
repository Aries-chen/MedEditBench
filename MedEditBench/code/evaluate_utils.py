from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def compute_rougeL(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    f = scores['rougeL'].fmeasure  
    return round(f, 3)  


def compute_bleu_score(text1, text2):
    hypothesis = text1.split()
    reference = text2.split()
    references = [reference]
    chencherry = SmoothingFunction()
    score = sentence_bleu(references, hypothesis, smoothing_function=chencherry.method2)
    return round(score, 3)
