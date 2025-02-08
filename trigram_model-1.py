import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Triyasha Ghosh Dastidar - tg2936
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    # Fetching all the words for which count is greater than 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    
    #Error condition
    
    if n < 1:
        return []
    else:
        sequence = ["START"]*(n-1) + sequence + ["STOP"]
    n_gram = [tuple(sequence[i:i+n]) for i in range(len(sequence) - (n-1))]

    return n_gram


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        # Remarks: This also marks the words which have word count < 1 to 'UNK'
        generator = corpus_reader(corpusfile, self.lexicon)
        self.vocab = len(self.lexicon)
        self.total_word_tokens = 0
        self.total_sentences = 0
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        # Initialize the unigram, bigram and trigram counts using defaultdict
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        words = 0
        sentences = 0

        ##Your code here
        for sequence in corpus:
            unigrams = get_ngrams(sequence,1)
            for unigram in unigrams:
                self.unigramcounts[unigram] +=1
            bigrams = get_ngrams(sequence,2)
            for bigram in bigrams:
                self.bigramcounts[bigram] +=1
            trigrams = get_ngrams(sequence,3)
            for trigram in trigrams:
                self.trigramcounts[trigram] +=1
            words += len(sequence)
            sentences +=1
                    
        self.total_word_tokens = words+sentences  # Taking into account "STOP"; one "STOP" per sentence
        self.total_sentences += sentences

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # Assume, given trigram = ('START', 'I', 'am')
        trigram_counts = self.trigramcounts[trigram]
        bigram = trigram[:2]
        bigram_counts = self.bigramcounts[bigram]
        if trigram_counts and bigram_counts:
            return trigram_counts/bigram_counts
        elif bigram_counts==0:
            return 1/self.vocab
        return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # Assume, given bigram = ('START', 'I')
        bigram_counts = self.bigramcounts[bigram]
        unigram = bigram[:1]
        if(unigram == ('START',)):
            return bigram_counts/self.total_sentences

        unigram_counts = self.unigramcounts[unigram]

        if bigram_counts and unigram_counts:
            return bigram_counts/unigram_counts
        elif unigram_counts==0:
            return 1/self.vocab
        return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # Assume, given unigram = ['START']
        unigram_counts = self.unigramcounts[unigram]
        if unigram_counts:
            return unigram_counts/self.total_word_tokens
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return 0.0
  
    def generate_sentence(self, t=20): 
        """
        Generates a random sentence from the trigram model. 
        t specifies the max length, but the sentence may be shorter if STOP is reached.
        """

        # Start with initial context ('START', 'START')
        current_bigram = ('START', 'START')
        sentence = []

        for _ in range(t):
            possible_trigrams = [trigram for trigram in self.trigramcounts 
                                if trigram[:2] == current_bigram]
            if not possible_trigrams:
                break

            trigram = random.choice(possible_trigrams)
            next_word = trigram[2]

            if next_word == 'STOP':
                break

            sentence.append(next_word)

            current_bigram = (current_bigram[1], next_word)

        return ' '.join(sentence)

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        # Assume trigram = ('START', 'I', 'am')
        w1, w2, w3 = trigram

        # Calculate unigram, bigram, and trigram probabilities
        unigram_prob = self.raw_unigram_probability((w3,))
        bigram_prob = self.raw_bigram_probability((w2, w3))
        trigram_prob = self.raw_trigram_probability(trigram)

        interpolated_prob = (lambda1 * unigram_prob) + (lambda2 * bigram_prob) + (lambda3 * trigram_prob)

        return interpolated_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        # Calculate the n-grams for the sentence
        trigrams = get_ngrams(sentence,3)
        final_probability = 0.0
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            if prob > 0:
                final_probability +=math.log2(prob)
            else:
                return float('-inf')
        return final_probability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the perplexity of the corpus.
        """
        total_corpus_log_prob = 0.0
        total_test_word_tokens = 0
        sent_count = 0
        for sentence in corpus:
            log_prob = self.sentence_logprob(sentence)
            total_corpus_log_prob += log_prob
            total_test_word_tokens += len(sentence)
            sent_count +=1
        total_test_word_tokens += sent_count # Include one "STOP" for each sentence Ref: https://edstem.org/us/courses/63584/discussion/5319763

        if total_test_word_tokens > 0:
            perplexity = 2 ** (-total_corpus_log_prob / total_test_word_tokens)
        else:
            perplexity = float('inf')
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp_high<pp_low:
                correct +=1
            total +=1


        for f in os.listdir(testdir2):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp_low<pp_high:
                correct +=1
            total +=1
        
        return correct / total if total > 0 else 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    #Testing out some scenarios
    # print(len(model.unigramcounts))
    # print(len(model.bigramcounts))
    # print(len(model.trigramcounts))
    # print(model.unigramcounts[('START',)])
    # print(model.total_word_tokens)
    # print(model.raw_bigram_probability(('START', 'the')))
    # print(model.sentence_logprob(['START', 'START', 'the']))
    
    # # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # # Essay scoring experiment: 
    # acc = essay_scoring_experiment("./hw1_data/ets_toefl_data/train_high.txt", "./hw1_data/ets_toefl_data/train_low.txt", "./hw1_data/ets_toefl_data/test_high", "./hw1_data/ets_toefl_data/test_low")
    # print(acc)

