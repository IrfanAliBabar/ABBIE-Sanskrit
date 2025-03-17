import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Define IAST character lists
vowel_iast = ['a', 'ā', 'i', 'ī', 'u', 'ū', 'e', 'o',]
consonant_iast = ['k', 'kh', 'g', 'gh', 'ṅ',
                  'c', 'ch', 'j', 'jh', 'ñ',
                  'ṭ', 'ṭh', 'ḍ', 'ḍh', 'ṇ',
                  't', 'th', 'd', 'dh', 'n',
                  'p', 'ph', 'b', 'bh', 'm',
                  'y', 'r', 'l', 'v', 'ś', 'ṣ', 's', 'h', 'ḷ', '|']
others_iast = ['ṃ', 'ḥ', 'ṛ', 'ṝ', 'ṁ', '~']

iast_charlist = vowel_iast + consonant_iast + others_iast

maxcompoundlen = 50
sandhi_window = 4

# Define transliteration mappings for IAST to SLP1 for better optimization and computation analysis
iast_to_slp1_map = {
    'a': 'a', 'ā': 'A', 'i': 'i', 'ī': 'I', 'u': 'u', 'ū': 'U', 'e': 'e', 'ai': 'E', 'o': 'O',
     'kh': 'K', 'kṣ':'KZ', 'gh': 'G', 'ṅ': 'N',
    'c': 'c', 'ch': 'C', 'j': 'j', 'jh': 'J', 'ñ': 'Y',
    'ṭ': 'w', 'ṭh': 'W', 'ḍ': 'q', 'ḍh': 'Q', 'ṇ': 'R', 'k': 'k','g': 'g',
    't': 't', 'th': 'T', 'd': 'd', 'dh': 'D', 'n': 'n',
    'p': 'p', 'ph': 'P', 'b': 'b', 'bh': 'B', 'm': 'm',
    'y': 'y', 'r': 'r', 'l': 'l', 'v': 'V', 'ś': 'S', 'ṣ': 'z', 's': 's', 'h': 'h', 'ḷ': 'x', '|': '|',
    'ṃ': 'M', 'ḥ': 'H', 'ṛ': 'f', 'ṝ': 'F', 'ṁ': 'Z', '~': '~'
}

def iast_to_slp1(text):
    slp1_text = ''.join([iast_to_slp1_map.get(char, '') for char in text])
    return slp1_text

def remove_noniast_chars(word):
    newword = ''
    for char in word:
        if char in iast_charlist:
            newword += char
    return newword

def get_sanskrit_dataset(datafile):
    datalist = []
    total = 0
    maxlen = 0
    minlen = float('inf')
    count = {}

    data = pd.read_csv(datafile)
    #print(f"Loaded {len(data)} rows from {datafile}")

    for index, row in data.iterrows():
        compound = row.get('compound')
        split = row.get('split')

        if not isinstance(compound, str) or not isinstance(split, str):
            continue  # Skip if not valid strings
        #print(f"Processing row {index}: compound={row['compound']}, split={row['split']}")
        compound = row['compound'].strip()
        split = row['split'].strip()

        words = split.split('+')

        if len(words) != 2:
            continue

        word1 = words[0].strip()
        slp1word1 = iast_to_slp1(remove_noniast_chars(word1))

        word2 = words[1].strip()
        slp1word2 = iast_to_slp1(remove_noniast_chars(word2))

        expected = compound.strip()
        slp1expected = iast_to_slp1(remove_noniast_chars(expected))

        if slp1word1 and slp1word2 and slp1expected:
            total += 1

            full_word_len = 2
            short_word_len = 2

            start = 0
            end = len(slp1expected)

            fullslp1expected = slp1expected
            fullslp1word1 = slp1word1
            fullslp1word2 = slp1word2

            if len(slp1word1) > full_word_len:
                start = len(slp1word1) - full_word_len
            if len(slp1word2) > short_word_len:
                end = end - len(slp1word2) + short_word_len

            var_length = len(slp1expected) - (len(slp1word1) + len(slp1word2))
            #print(f"slp1_words: {slp1word1}, slp1_expected: {slp1expected}, var_length: {var_length}")

            if var_length < 2 and var_length > -3 and len(slp1expected) > len(slp1word1) and len(slp1expected) > len(slp1word2) and len(fullslp1expected) <= maxcompoundlen and len(fullslp1expected) >= sandhi_window:

                startblock = False
                endblock = False

                while end - start < sandhi_window:
                    if start > 0:
                        start = start - 1
                    else:
                        startblock = True
                    if end - start == sandhi_window:
                        break
                    if end < len(slp1expected):
                        end = end + 1
                    else:
                        endblock = True
                    if end - start == sandhi_window:
                        break
                    if startblock and endblock:
                        break

                newlen = len(slp1word2) - len(slp1expected) + end
                #print(f"Before appending: start={start}, end={end}, fullslp1words={fullslp1expected}, expected={slp1expected[start:end]}")
                #print(f"Comparing: expected={slp1expected}, fullslp1words={fullslp1expected}, start={start}, end={end}, expected[start:end]={slp1expected[start:end]}")

                if slp1word1[:start] == slp1expected[:start] and slp1word2[newlen:] == slp1expected[end:]:
                    slp1word1 = slp1word1[start:]
                    slp1word2 = slp1word2[:newlen]
                    slp1expected = slp1expected[start:end]

                    datalist.append([slp1word1, slp1word2, slp1expected, fullslp1expected, int(start), int(end), fullslp1word1, fullslp1word2])

                    if (maxlen < end - start):
                        maxlen = end - start
                    if (minlen > end - start):
                        minlen = end - start

                    if (end - start) in count:
                        count[end - start] += 1
                    else:
                        count[end - start] = 1
        #print(f"Total valid entries in dataset: {len(datalist)}")

    return datalist

def prepare_data(datafile):
    dl = get_sanskrit_dataset(datafile)
    return dl
