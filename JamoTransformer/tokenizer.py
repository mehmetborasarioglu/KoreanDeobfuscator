"""
JamoTokenizer: A Korean tokenizer that decomposes Hangul syllables into Jamos.
- Preserves non-Korean characters (e.g., numbers, punctuation)
- Encodes text at the sub-character level (초성, 중성, 종성)
- Supports special tokens for seq2seq training

Requires: jamo (pip install jamo)
"""

from jamo import h2j, j2h
from collections import defaultdict

# Unicode-safe syllable checker
def is_hangul_syllable(char):
    return '\uAC00' <= char <= '\uD7A3'

class JamoTokenizer:
    def __init__(self, texts, special_tokens=["<pad>", "<s>", "</s>", "<unk>"]):
        jamo_set = set()

        #Delete this loop and instead create vocab from only jamo
        for line in texts:
            for char in line.strip():
                if is_hangul_syllable(char):
                    jamo_seq = h2j(char)
                    jamo_set.update(jamo_seq)
                else:
                    jamo_set.add(char)

        self.special_tokens = special_tokens
        self.vocab = sorted(jamo_set)
        self.token2id = {tok: i for i, tok in enumerate(special_tokens + self.vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        self.pad_token_id = self.token2id["<pad>"]
        self.bos_token_id = self.token2id["<s>"]
        self.eos_token_id = self.token2id["</s>"]
        self.unk_token_id = self.token2id["<unk>"]

    def encode(self, text, max_length):
        jamos = []
        for c in text.strip():
            if is_hangul_syllable(c):
                jamos.extend(h2j(c))
            else:
                jamos.append(c)
        ids = [self.token2id.get(j, self.unk_token_id) for j in jamos]
        ids = [self.bos_token_id] + ids[:max_length - 2] + [self.eos_token_id]
        padding = [self.pad_token_id] * (max_length - len(ids))
        return ids + padding

    def decode(self, ids):
        # convert ids into tokens and ignore special tokens.
        tokens = [self.id2token.get(i, "") for i in ids if self.id2token.get(i, "") not in self.special_tokens]
        result = []
        buffer = []

        # helper functions to classify jamo types.
        def is_initial(j):
            # initial consonants in the jamo block (e.g. U+1100 - U+1112)
            return 0x1100 <= ord(j) <= 0x1112

        def is_medial(j):
            # medial vowels (e.g. U+1161 - U+1175)
            return 0x1161 <= ord(j) <= 0x1175

        def is_final(j):
            # final consonants (typically U+11A8 - U+11C2)
            return 0x11A8 <= ord(j) <= 0x11C2

        def flush_buffer():
            nonlocal buffer
            if buffer:
                if len(buffer) == 2:
                    # two-element syllable (initial, medial) with an empty final.
                    try:
                        result.append(j2h(buffer[0], buffer[1], ''))
                    except Exception:
                        result.extend(buffer)
                elif len(buffer) == 3:
                    # three-element syllable (initial, medial, final).
                    try:
                        result.append(j2h(buffer[0], buffer[1], buffer[2]))
                    except Exception:
                        result.extend(buffer)
                else:
                    result.extend(buffer)
            buffer = []

        for token in tokens:
            # if the token is not part of the Hangul jamo ranges, flush any buffered jamos and append the token.
            if not (is_initial(token) or is_medial(token) or is_final(token)):
                flush_buffer()
                result.append(token)
                continue

            # process Hangul jamos.
            if not buffer:
                # start a new potential syllable.
                buffer.append(token)
            else:
                if len(buffer) == 1:
                    # expect a medial vowel to follow an initial.
                    if is_medial(token):
                        buffer.append(token)
                    else:
                        # if not a medial, flush current buffer and start new.
                        flush_buffer()
                        buffer.append(token)
                elif len(buffer) == 2:
                    # we have an initial and medial; now check if token is a valid final.
                    if is_final(token):
                        buffer.append(token)
                        flush_buffer()  # Once complete, flush the syllable.
                    else:
                        # otherwise, complete the syllable without a final and start a new one.
                        flush_buffer()
                        buffer.append(token)
                else:
                    # in case buffer length is not 1 or 2, flush it and start fresh.
                    flush_buffer()
                    buffer.append(token)

        flush_buffer()
        return ''.join(result)

    def vocab_size(self):
        return len(self.token2id)

# Example usage
if __name__ == "__main__":
    samples = ["견환123!", "초반asdasdاسداففف부에는~", "점심을 먹으러 갑시다.","씬뎨렐랴 플료젝특위 맏찜먁 윤닛읏료 뒈콜롑윅선 익휴 뗍퓜 얘청있엇곶, 곡많 많둘교 않찍 윤닛멍은 졍헤찌치 않햐써 『＊』 문짢 한낢많 써놓교 귀훽셜룰 쟉셩햐교 윗섯는테, 맒찜 쟈겹토쭝 뤽캄와 밉뤼약캅 둘럿왁써 앉위돎 펫쑥끼훽갸늚 포는 빨람뭬 뚫과 잉얍키 한늚랴 윤닛 위룸 짇눈껸 낯중 읾뤼 퇴옅꼬, 값짝쓺럽께 믹큐와 뤼임낢위 뎁뷔뮷텝갸 쟈퓌차 굽팡께 윙뜰만웨 씽굶 탸일틂꼭케 캇샬룰 뿟엇셩 읾단 툴룰 욜렵뵤냇타."]
    tokenizer = JamoTokenizer(samples)
    for sample in samples:    
        encoded = tokenizer.encode(sample, max_length=1024)
        print()
        print(sample)
        print("Encoded:", encoded)
        print("Decoded:", tokenizer.decode(encoded))
        print()
    print("Vocab size:", tokenizer.vocab_size())