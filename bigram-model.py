import math


class BigramModel:

    def __init__(self):
        self.word2count = {}
        self.totalWordCount = 0
        self.bigrams2count = {}
        self.lastWord = None

    def addWord(self, word):
        if not word in self.word2count:
            self.word2count[word] = 0
            self.totalWordCount += 1
        else:
            self.word2count[word] += 1

        if self.lastWord:
            bigram = (self.lastWord, word)
            print(bigram)
            if not bigram in self.bigrams2count:
                self.bigrams2count[bigram] = 0
            else:
                self.bigrams2count[bigram] += 1

        self.lastWord = word

    def getWordCount(self, word):
        wordCount = 0
        if word in self.word2count.keys():
            wordCount = self.word2count[word]
        return wordCount

    def addSentence(self, sentence):
        self.lastWord = None
        for word in sentence.lower().split():
            self.addWord(word)

    def propability(self, sentence):
        formatedSentence = sentence.lower().split()
        probability = math.log(self.getWordCount(formatedSentence[0]) + 1 / self.totalWordCount)

        self.lastWord = formatedSentence[0]

        for wordIndex in range(len(formatedSentence) - 1):
            word = formatedSentence[wordIndex + 1]
            bigramCount = 0
            if (self.lastWord, word) in self.bigrams2count.keys():
                bigramCount = self.bigrams2count[(self.lastWord, word)]

            probability += math.log(bigramCount + 1 / (self.getWordCount(word) + self.totalWordCount))
            self.lastWord = word

        return (1.0 / self.totalWordCount) * (probability)


b = BigramModel()
b.addSentence("The quick brown fox jumps over the lazy dog")

print()
print(b.propability("The quick brown fox jumps over the lazy dog"))
print(b.propability("The quick brown fox jumps over the lazy turtle"))
print(b.propability("The quick brown fox jumps over the crazy turtle"))
print(b.propability("The quick yellow fox jumps over the crazy turtle"))
print(b.propability("a asdf nnn fff bbb mm eee crhhhazy turmmmtle"))
print(b.propability("The the the"))
print(b.propability("brown fox"))