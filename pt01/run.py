"""Работа 1 заочного отделения. Задание: part-time/assignments/01/README.md в репозитории курса.

Запуск: python pt01/run.py, для CI python pt01/run.py --fast.
Артефакты bigram.json и bpe.json пишутся в pt01/artifacts/.
"""
import argparse
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict  # для работы со словарями
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
)
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
SEED = 42

END = "</w>"  # маркер конца слова (для BPE - Byte-Pair Encoding)


# ======= Корпус =======
def download_corpus() -> None:
    # скачиваем корпус (Война и мир)
    if not CORPUS.exists():
        print(f"Скачивание корпуса в {CORPUS} ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS)


def load_words() -> list[str]:
    # приводим к нижему регистру, заменяем ё на е
    text = CORPUS.read_text(encoding="utf-8").lower().replace("ё", "е")
    # выделяем слова регулярным выражением
    return WORD_RE.findall(text)


# ======= Биграммная модель =======
class BigramLM:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.unigram: Counter = Counter()  # словарь. сколько раз встретилось каждое слово v
        # словарь биграм. ключ - первое слово пары v,
        # значение - словарь (ключ - второе слово w, значение - сколько раз пара (v, w) встретилась)
        # defaultdict(Counter) автоматически создает  пустой объект Counter в качестве значения
        # при обращении к несуществующему ключу
        self.bigram: dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()  # множество уникальных слов
        self.V = 0  # размер словаря

    def fit(self, words: list[str]) -> None:
        self.unigram = Counter(words)  # словарь. сколько раз встретилось каждое слово v
        self.vocab = set(self.unigram)  # множество уникальных слов. счетчик слов отбрасываем
        self.V = len(self.vocab)
        self.bigram = defaultdict(Counter)
        # считаем, сколько раз встречались пары
        for v, w in zip(words, words[1:]):
            self.bigram[v][w] += 1

    def prob(self, v: str, w: str) -> float:
        c_vw = self.bigram[v][w] if v in self.bigram else 0  # сколько раз встречалась пара v w
        c_v = self.unigram.get(v, 0)  # сколько раз встречалось слово v
        # вероятность следующего слова (со сглаиванием Лапласса)
        return (c_vw + self.alpha) / (c_v + self.alpha * self.V)

    def logprob(self, words: list[str]) -> float:
        # считаем правдоподобие текста
        if len(words) < 2:
            return 0.0
        return sum(
            math.log(self.prob(v, w)) for v, w in zip(words, words[1:])
        )

    def next_distribution(self, v: str, temperature: float) -> list[tuple[str, float]]:
        # Создание списка кандидатов: слова из словаря + вероятности
        base = self.bigram.get(v)  # получаем список биграм для слова v
        if base:
            candidates = list(base.keys())  # получаем список кандидатов
        else:
            # ни разу не видели контекст - берём top-k униграмм
            candidates = [w for w, _ in self.unigram.most_common(20)]
        probs = []
        for w in candidates:  # перебираем кандидатов и рассчитываем их вероятности
            p = self.prob(v, w)
            probs.append(p ** (1.0 / temperature))  # расчет вероятности идет с учетом температуры
            # когд температура блзко к 0 - модель будет максимально строгой и предсказуемой
            # когд температура блзко к 1 - модель может выбирать менее вероятные слова, текст "интереснее"
        # нормализация вероятностей (чтобы сумма была равна 1)
        s = sum(probs)
        probs = [p / s for p in probs]
        return list(zip(candidates, probs))

    def generate(self, start: str, n: int, temperature: float, rng: random.Random) -> list[str]:
        # генерация текста
        out = [start]
        cur = start
        for _ in range(n - 1):
            dist = self.next_distribution(cur, temperature)  # получаем список кандидатов
            words = [w for w, _ in dist]
            probs = [p for _, p in dist]
            nxt = rng.choices(words, weights=probs, k=1)[0]
            # взвешенное сэмплирование с возвращением.
            # Выбирает k элементов из population, где вероятность выбрать элемент i пропорциональна weights[i].
            # population = words - кандидаты, weights - их вероятности, k=1 - выбираем одно слово
            out.append(nxt)
            cur = nxt  # переходим к следующему слову
        return out


def perplexity(model: BigramLM, words: list[str]) -> float:
    # расчет перплексии
    if len(words) < 2:
        return float("inf")
    total = sum(math.log(model.prob(v, w)) for v, w in zip(words, words[1:]))
    return math.exp(-total / (len(words) - 1))


#  ======= BPE =======
def word_to_chars(word: str) -> tuple[str, ...]:
    # переводим слово в список символов и добавляем маркер конца слова
    return tuple(list(word) + [END])


def train_bpe(word_counts: Counter, num_merges: int) -> list[tuple[str, str]]:
    # словарь: слово - текущее разбиение на символы
    vocab: dict[str, list[str]] = {
        w: list(word_to_chars(w)) for w in word_counts
    }
    merges: list[tuple[str, str]] = []  # список с парами для слияния
    for _ in range(num_merges):
        pair_counts: Counter = Counter()  # создаем словарь-счетчик пар
        for w, chars in vocab.items():
            freq = word_counts[w]  # сколько раз встречалось слово в тексте
            for a, b in zip(chars, chars[1:]):
                pair_counts[(a, b)] += freq  # сколько раз пара встречалась в тексте
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]  # берем одну самую частую пару
        # ([0] так как возвращается массив (в данном случае с одним элементом) пар)
        if best_count < 2:  # если самая частая пара встречается меньше 2 раз, сливать нечего
            break
        merges.append(best_pair)
        a, b = best_pair
        new_tok = a + b  # проводим слияние лучшей пары
        # проходим по всем словам и заменяем все вхождения пары a+b на символ ab
        for w, chars in vocab.items():  # перебираем все слова
            if a in chars:  # перебираем символы слова
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        # если нашли пару a+b - заменяем на символ ab и добавляем в список символов
                        new_chars.append(new_tok)
                        i += 2
                    else:
                        # иначе просто добавляем следующий символ в список
                        new_chars.append(chars[i])
                        i += 1
                vocab[w] = new_chars
    return merges


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    # ранги: пара - ее номер в списке
    ranks = {pair: i for i, pair in enumerate(merges)}
    chars = list(word_to_chars(word))  # разбиваем слово на символы
    while len(chars) > 1:  # пока не закончились пары символов для слияния в слове
        best_rank = None
        best_idx = -1
        for i in range(len(chars) - 1): # перебираем пары
            pair = (chars[i], chars[i + 1])
            r = ranks.get(pair)
            # если такая пара есть в списке слияний и ранг меньше best_rank,
            # то этот ранг становится лучшим
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_idx = i
        if best_rank is None:  # если не нашли ни одного слияния - выходим из цикла
            break
        a, b = merges[best_rank]
        chars = chars[:best_idx] + [a + b] + chars[best_idx + 2:]  # сливаем найденную пару
    return chars


# ======= Функции для запуска обработки =======
def build_bigram(all_words: list[str]) -> dict:
    n = len(all_words)
    split = int(n * 0.9)
    train_words = all_words[:split]
    val_words = all_words[split:]

    model = BigramLM(alpha=0.01)
    model.fit(train_words)

    top20 = [[w, c] for w, c in model.unigram.most_common(20)]

    # perplexity на отложенной части
    ppls = {}
    for alpha in (1.0, 0.1, 0.01):
        m = BigramLM(alpha=alpha)
        m.fit(train_words)
        # условие только чтобы в json 1 записывалось без точки (1, а не 1.0)
        ppls[str(alpha) if alpha != 1 else "1"] = perplexity(m, val_words)

    # правдоподобие
    first12 = val_words[:12]
    shuffled = first12[:]
    random.Random(SEED).shuffle(shuffled)  # перемешиваем первые 12 слов
    lp_sent = model.logprob(first12)
    lp_shuf = model.logprob(shuffled)
    # считаем, сколько пар не встречалось в обучеющей части
    unseen = sum(
        1 for v, w in zip(first12, first12[1:])
        if model.bigram.get(v, Counter()).get(w, 0) == 0
    )

    # генерация
    generation = []
    for T in (0.5, 1.0, 2.0):  # для каждой температуры
        for start in ("князь", "она"):
            text = model.generate(start, 30, T, random.Random(SEED))
            generation.append({
                "temperature": T, # температура
                "start": start,  # начальное слово
                "text": " ".join(text),  # полученный текст
            })

    return {
        "corpus": CORPUS.name,  # имя файла с корпусом
        "words": n,  # число слов в корпусе
        "train_words": len(train_words),  # число слов в "обучающей" выборке
        "val_words": len(val_words),  # число слов в "тестовой" выборке
        "vocab_size": model.V,  # размер словаря (число уникальных слов)
        "top20": top20,  # 20 самых популярных слов (слово - счетчик)
        "perplexity": ppls,  # перплексия (для разных альфа из формулы сглаживания Лапласса)
        "likelihood": {
            "sentence": " ".join(first12),
            "shuffled": " ".join(shuffled),
            "alpha": 0.01,
            "logprob_sentence": lp_sent,  # правдоподобие исходного предложения
            "logprob_shuffled": lp_shuf,  # правдоподобие перемешанного предложения
            "unseen_pairs_in_sentence": unseen,  # сколько пар не встречалось в обучающей части
        },
        "generation": generation,  # результаты генерации текстов
    }


def build_bpe(all_words: list[str], fast: bool) -> dict:
    num_merges = 200 if fast else 1500
    word_counts = Counter(all_words)
    # порог для ускорения: слова с частотой ниже порога не берём в обучение
    threshold = 2 if not fast else 3
    train_counts = Counter({w: c for w, c in word_counts.items() if c >= threshold})
    if not train_counts:
        train_counts = word_counts

    print(f"BPE: обучение {num_merges} слияний на {len(train_counts)} словах")
    merges = train_bpe(train_counts, num_merges)

    # статистика на всём корпусе
    chars_total = 0  # общее число символов
    words_total = len(all_words)  # общее число слов
    tokens_total = 0  # общее число токенов
    examples = {}
    wanted = {"наполеон", "сказала", "неудовольствие"}
    for w, c in word_counts.items():  # к каждому слову применяем слияния (BPE)
        toks = apply_bpe(w, merges)
        tokens_total += len(toks) * c  # c - сколько раз слово встретилось в тексте
        chars_total += len(w) * c
        if w in wanted:
            examples[w] = toks  # записываем примеры

    return {
        "num_merges": len(merges),  # число слияний
        "vocab_size": len(train_counts) + sum(len(m) for m in merges) + 1,
        "merges": [list(p) for p in merges],  # слияния
        "examples": examples,  # примеры слияний для слов наполеон, сказала, неудовольствие
        "stats": {
            "chars": chars_total,  # общее число символов
            "words": words_total,
            "tokens_bpe": tokens_total,
            # среднее число токенов на слово
            "tokens_per_word": tokens_total / words_total if words_total else 0.0,
            # во сколько раз токенов меньше, чем букв
            "compression_vs_chars": chars_total / tokens_total if tokens_total else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    download_corpus()  # загружаем корпус
    # обрабатываем текст (к нижнему регистру, ё->е, выбираем слова регулярным выражением)
    all_words = load_words()
    print(f"Слов в корпусе: {len(all_words)}")

    print("Строим биграммную модель")
    bigram = build_bigram(all_words)
    (ARTIFACTS / "bigram.json").write_text(
        json.dumps(bigram, ensure_ascii=False, indent=2), encoding="utf-8"
    )  # ensure_ascii=False чтобы русский текст был читаемым
    # indent = 2 - "красивое" форматирование с отступами в 2 пробела и переносами на новые строки

    print("Обучаем BPE")
    bpe = build_bpe(all_words, args.fast)
    (ARTIFACTS / "bpe.json").write_text(
        json.dumps(bpe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Готово. Результаты в", ARTIFACTS)


if __name__ == "__main__":
    main()
