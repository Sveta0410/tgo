"""Работа 2 заочного отделения. Задание: part-time/assignments/02/README.md в репозитории курса.

Запуск: python pt02/run.py, для CI python pt02/run.py --fast.
Слияния BPE берутся из pt01/artifacts/bpe.json.
"""
import argparse
from pathlib import Path
import csv
import json
import math
import re
import time
import urllib.request
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
)
BPE = HERE.parent / "pt01" / "artifacts" / "bpe.json"
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
END = "</w>"
SEED = 42


# ======= Данные и BPE =======
def download_corpus() -> None:
    if not CORPUS.exists():
        print(f"Скачивание корпуса в {CORPUS} ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS)


def load_words() -> list[str]:
    # приводим к нижнему регистру, заменяем ё на е
    text = CORPUS.read_text(encoding="utf-8").lower().replace("ё", "е")
    # выделяем слова регулярным выражением
    return WORD_RE.findall(text)


def word_to_chars(word: str) -> list[str]:
    # переводим слово в список символов и добавляем маркер конца слова
    return list(word) + [END]


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    # ранги: пара - ее номер в списке
    ranks = {pair: i for i, pair in enumerate(merges)}
    chars = word_to_chars(word)  # разбиваем слово на символы
    while len(chars) > 1:  # пока не закончились пары символов для слияния в слове
        best_rank, best_idx = None, -1
        for i in range(len(chars) - 1):  # перебираем пары
            r = ranks.get((chars[i], chars[i + 1]))
            # если такая пара есть в списке слияний и ранг меньше best_rank,
            # то этот ранг становится лучшим
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_idx = r, i
        if best_rank is None:  # если не нашли ни одного слияния - выходим из цикла
            break
        a, b = merges[best_rank]
        chars = chars[:best_idx] + [a + b] + chars[best_idx + 2:]  # сливаем найденную пару
    return chars


def load_bpe(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(a, b) for a, b in data["merges"]]


def build_vocab(all_words: list[str], merges: list[tuple[str, str]]):
    # Строим словарь токенов и поток id.
    # Словарь: токен-id. </w> получает id 0. остальные добавляются по мере встречи
    token_to_id: dict[str, int] = {END: 0}
    ids: list[int] = []  # список номеров токенов
    word_lengths: list[int] = []  # сколько токенов в каждом слове
    for w in all_words:  # проходим по всем словам
        toks = apply_bpe(w, merges)  # делим слова на токены
        # добавляем новые токены в словарь по мере встречи
        for t in toks:
            if t not in token_to_id:
                token_to_id[t] = len(token_to_id)
        # для каждого токена в слове добавляем его id в ids
        ids.extend(token_to_id[t] for t in toks)
        word_lengths.append(len(toks))
    return token_to_id, ids, word_lengths


# ======= Модель: ручное многоголовое внимание =======
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model  # ширина модели (размерность вектора признаков для каждого токена)
        self.n_heads = n_heads  # число голов (каждая голова учится смотреть на что-то своё)
        self.d_head = d_model // n_heads  # ширина одной головы
        # линейный слой, который из (B, T, 128) делает (B, T, 3*128). он вычисляет Q, K, V - три вектора на каждый токен
        # Query (запрос), Key (ключ), Value (значение)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)  # выходной линейный слой
        # матрица (128, 128). torch.triu оставляет верхний треугольник выше диагонали
        # (там значения True - запрещено смотреть), остальные значения False
        mask = torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1)
        # регистрирует тензор как буфер: он не обучается, но переносится с моделью (.to(device)) и сохраняется
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B - размер батча (число окон), T - длина окна, D - d_model, размерность вектора признаков для каждого токена
        B, T, D = x.shape
        # для каждого токена вектор из 384 чисел: первые 128- запрос (Q), вторые 128- ключ (K), третьи 128- значение (V)
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # q, k, v — отдельные тензоры

        # (B, T, D) -> (B, heads, T, d_head)
        # разбиваем вектор длины 128 на 4 головы по 32 (32, 128, 128) → (32, 128, 4, 32)
        # потом меняем оси 1 и 2: (32, 128, 4, 32) → (32, 4, 128, 32) (transpose(1, 2))
        # T, d_head в конце так как по ним идут матричные операции
        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        # ядро внимания
        # q @ kᵀ - «насколько токен i хочет смотреть на токен j»
        # делим на √32 чтобы softmax не давал только одну долю близкую к единице, а остальные около нуля
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, heads, T, T)
        # где mask истинна (верхний треугольник), ставим −∞
        # после softmax exp(−∞) = 0 - запрещённые позиции получают нулевой вес
        # (чтобы модель не смогла "смотреть в будущее")
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)  # scores становится распределением вероятностей
        ctx = attn @ v  # взвешенная сумма значений
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)  # склейка голов обратно
        return self.out(ctx)  # Выходной линейный слой (cмешивает головы, чтобы каждая координата зависела от всех.)
        # (Без него головы были бы «параллельными каналами» без взаимодействия.)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, context: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # LayerNorm перед вниманием (pre-norm) (чтобы обучение было устойчивее)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, context)  # многоголовое внимание
        self.ln2 = nn.LayerNorm(d_model)  # нормализация
        self.ff = nn.Sequential(  # полносвязная сеть
            nn.Linear(d_model, d_ff),  # 128 -> 512
            nn.GELU(),  # нелинейность (Gaussian Error Linear Unit)
            nn.Linear(d_ff, d_model),  # 512 -> 128
        )

    def forward(self, x):
        # attn - многоголовое внимание. токены обмениваются информацией
        # x + - остаточная связь (прибавляем результат внимания к исходному x, чтобы информация не терялась)
        x = x + self.attn(self.ln1(x))
        # обработка каждого токена отдельно
        x = x + self.ff(self.ln2(x))
        return x


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, n_layers=4,
                 n_heads=4, d_ff=512, context=128):
        super().__init__()
        self.context = context  # максимальная длина контекста - окна токенов на входе
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # таблица эмбеддингов токенов
        self.pos_emb = nn.Embedding(context, d_model)  # таблица эмбеддингов позиций
        # 4 (n_layers) одинаковых блока
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, context) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)  # нормализация
        # bias=False, чтобы уменьшить число параметров (качество работы не изменится - сдвиг уже есть в нормализации)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # выходной линейный слой

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape  # B - размер батча (число окон), T - длина окна
        pos = torch.arange(T, device=idx.device)  # создаем тензор с номерами позиций от 0 до T-1 с шагом 1
        # self.tok_emb(idx) - (B, T, d_model). Для каждого id достаём вектор.
        # self.pos_emb(pos) - (T, d_model). Для каждой позиции достаём вектор.
        # [None] — добавляет ось: (1, T, d_model), чтобы сложить с (B, T, d_model)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for block in self.blocks:  # проходим через блоки
            x = block(x)
        x = self.ln_f(x)  # нормализация
        return self.head(x)  # (B, T, vocab)


# ======= Обучение =======
def get_batch(ids: torch.Tensor, batch: int, context: int, gen: torch.Generator):
    n = ids.numel()  # numel() - количество элементов в тензоре
    # берем batch батчей по context токенов из случайных мест обучающей части
    starts = torch.randint(0, n - context - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + context] for s in starts])  # то, что модель видит
    y = torch.stack([ids[s + 1:s + context + 1] for s in starts])  # то, что модель должна будет предсказать
    return x, y


@torch.no_grad()  # отключаем построение графа вычислений, так как модель мы тут не тренируем
def eval_loss(model, ids, val_starts, context):
    model.eval()  # переключаем модель в режим оценки
    total, count = 0.0, 0  # total - сумма −log P по всем позициям, count - сколько позиций оценено
    for s in val_starts:  # Проходим по стартовым позициям
        x = ids[s:s + context][None]  # то, что модель видит
        y = ids[s + 1:s + context + 1][None]  # то, что должна предсказать
        logits = model(x)  # Прямой проход. Для каждой позиции - распределение следующего токена
        loss = F.cross_entropy(  # вычисляем ошибку
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total += loss.item()
        count += y.numel()
    model.train()  # возвращаем модель в режим обучения
    return total / count  # средний лосс на токен


# генерация токенов
@torch.no_grad()  # отключаем построение графа вычислений, так как модель мы тут не тренируем
def generate_ids(model, prompt_ids: list[int], max_new: int,  # max_new - сколько токенов нужно сгенерировать
                 temperature: float = 1.0, top_p: float | None = None,
                 context: int = 128, gen: torch.Generator | None = None) -> list[int]:
    model.eval()  # переключаем модель в режим оценки
    ids = list(prompt_ids)  # копируем начальные токены
    for _ in range(max_new):  # генерируем по 1 токену
        window = ids[-context:]  # берём последние 128 (context) токенов
        x = torch.tensor([window], dtype=torch.long)  # делаем из списка тензор (1, 128)
        logits = model(x)[0, -1]  # логиты для следующего токена - последняя позиция окна
        if temperature == 0:  # жадный режим - всё распределение схлопывается в одну точку, на самом вероятном токене
            nxt = int(torch.argmax(logits))  # иначе будет деление на ноль
        else:
            logits = logits / temperature  # делим логиты на температуру
            probs = F.softmax(logits, dim=-1)  # получаем вероятности
            # из распределения отбрасывается "хвост" маловероятных токенов, а выбор идёт только из "ядра"
            # (nucleus) — наименьшего набора токенов, чья суммарная вероятность достигает top_p (0.9)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)  # сортируем вероятности по убыванию
                # кумулятивная сумма = каждый элемент - сумма всех предыдущих включая себя
                cum = torch.cumsum(sorted_probs, dim=0)
                # оставляем наименьший набор с суммой >= p
                cutoff = int((cum >= top_p).nonzero()[0]) + 1  # первый индекс, где сумма ≥ top_p
                kept_idx = sorted_idx[:cutoff]  # отбрасываем лишнее
                kept_probs = sorted_probs[:cutoff]
                kept_probs = kept_probs / kept_probs.sum()  # перенормировка, чтобы сумма была = 1
                # multinomial выбирает случайный индекс в kept_probs по вероятностям
                # kept_idx - получаем id токена
                nxt = int(kept_idx[torch.multinomial(kept_probs, 1, generator=gen)])
            else:
                nxt = int(torch.multinomial(probs, 1, generator=gen))
        ids.append(nxt)  # добавляем токен
        if len(ids) > len(prompt_ids) + max_new:
            break
    model.train()  # возвращаем модель в режим обучения
    return ids[len(prompt_ids):]  # возвращаем только новые токены


def detokenize(ids: list[int], id_to_tok: dict[int, str]) -> str:
    # Склеиваем токены, </w> заменяем на пробел.
    parts = []
    for i in ids:
        t = id_to_tok.get(i, "")  # из id получаем токен
        if t == END:
            continue
        parts.append(t.replace(END, " "))  # если </w> внутри токена, заменяем на пробел
    return "".join(parts).strip()


def find_loop_start(ids: list[int], n: int = 4):
    # поиск зацикливания
    # возвращается позиция первой новой четвёрки, уже встречавшейся раньше, или None
    new = ids  # ids уже только новые
    seen = set()
    for i in range(len(new) - n + 1):
        quad = tuple(new[i:i + n])  # четвёрка подряд идущих токенов
        if quad in seen:
            return i
        seen.add(quad)
    return None


# ======= Биграммы (из первой работы) =======
class BigramLM:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.unigram: Counter = Counter()  # словарь. сколько раз встретилось каждое слово v
        # словарь биграм. ключ - первое слово пары v,
        # значение - словарь (ключ - второе слово w, значение - сколько раз пара (v, w) встретилась)
        # defaultdict(Counter) автоматически создает  пустой объект Counter в качестве значения
        # при обращении к несуществующему ключу
        self.bigram: dict[str, Counter] = defaultdict(Counter)
        self.V = 0  # размер словаря

    def fit(self, words):
        self.unigram = Counter(words)  # словарь. сколько раз встретилось каждое слово v
        self.V = len(self.unigram)
        self.bigram = defaultdict(Counter)
        # считаем, сколько раз встречались пары
        for v, w in zip(words, words[1:]):
            self.bigram[v][w] += 1

    def logprob(self, v: str, w: str) -> float:
        c_vw = self.bigram[v][w] if v in self.bigram else 0  # сколько раз встречалась пара v w
        c_v = self.unigram.get(v, 0)  # сколько раз встречалось слово v
        # считаем правдоподобие текста
        # логарифм от вероятности следующего слова (со сглаживанием Лапласса)
        return math.log((c_vw + self.alpha) / (c_v + self.alpha * self.V))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    ap.add_argument("--bpe", type=Path, default=BPE)
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    torch.manual_seed(SEED)  # фиксирует инициализацию весов
    gen = torch.Generator().manual_seed(SEED)  # генератор для всех случайных выборок в обучении
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()  # засекаем время начала выполнения кода

    download_corpus()
    all_words = load_words()
    merges = load_bpe(args.bpe)
    # vocab - словарь (токен-id), ids - список номеров токенов, word_lengths - сколько токенов в каждом слове
    vocab, ids, word_lengths = build_vocab(all_words, merges)
    id_to_tok = {i: t for t, i in vocab.items()}  # "переворачиваем" vocab, получаем словарь (id-токен)
    vocab_size = len(vocab)
    print(f"Слов: {len(all_words)}, токенов: {len(ids)}, словарь: {vocab_size}")

    ids_t = torch.tensor(ids, dtype=torch.long)  # переводим список в тензор

    # откладываем последние 10% слов для проверки
    split_word = int(len(all_words) * 0.9)
    split_tok = sum(word_lengths[:split_word])
    train_ids = ids_t[:split_tok]
    val_ids = ids_t[split_tok:]

    # параметры
    if args.fast:
        # батч - 16 окон, в каждом будет 128 токенов
        # log_every - каждые 25 шагов считаем лосс на обучающих окнах за прошедший отрезок
        # и лосс на двадцати фиксированных окнах из отложенной части
        steps, batch, log_every = 200, 16, 25
        sample_steps = [0, 50, 100, steps]
        # n_eval_gen для samples - образцы по шагам
        # n_gen для пункта "генерация руками"
        n_eval_gen, n_gen = 60, 40
        eval_words = 300
    else:
        steps, batch, log_every = 2000, 32, 100
        sample_steps = [0, 100, 500, steps]
        n_eval_gen, n_gen = 60, 80
        eval_words = 2000
    lr = 1e-3  # learning rate - шаг обучения
    context = 128  # число токенов в окне

    # двадцать фиксированных окон из отложенной части (для расчета потерь (loss) каждые 100 (25 в fast) шагов)
    n_val_windows = 20
    # numel() - количество элементов в тензоре
    max_start = val_ids.numel() - context - 1
    # val_starts - индексы в тензоре val_ids, откуда мы начинаем "срезать" окно
    val_starts = [int(torch.randint(0, max_start, (1,), generator=gen)) for _ in range(n_val_windows)]

    model = TinyDecoder(vocab_size, context=context).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # токенизируем prompt
    prompt_text = "князь андрей"
    prompt_ids = []
    for w in WORD_RE.findall(prompt_text.lower().replace("ё", "е")):
        for t in apply_bpe(w, merges):  # делим слово на токены
            prompt_ids.append(vocab.get(t, 0))  # получаем список id токенов промпта

    # ---- обучение + кривая + образцы по шагам ----
    curve = []
    samples = []
    # val_loss на случайных весах, до первого обновления (на отложенной части)
    step0_val = eval_loss(model, val_ids, val_starts, context)
    # train_loss на шаге 0 на обучающих окнах
    x0, y0 = get_batch(train_ids, batch, context, gen)
    with torch.no_grad():
        l0 = F.cross_entropy(
            model(x0.to(device)).view(-1, vocab_size), y0.to(device).view(-1)
        ).item()
    curve.append((0, l0, step0_val))  # запоминаем train_loss и val_loss на 0 шаге

    def do_sample(step):  # генерируем n_eval_gen новых токенов и переводим в текст
        ids_new = generate_ids(model, prompt_ids, n_eval_gen,
                               temperature=0.8, context=context, gen=gen)
        text = detokenize(ids_new, id_to_tok)
        samples.append({"step": step, "text": prompt_text + " " + text})

    if 0 in sample_steps:
        do_sample(0)

    print(f"Обучение: {steps} шагов, batch={batch}, device={device}")
    running, running_n = 0.0, 0
    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, batch, context, gen)
        x, y = x.to(device), y.to(device)
        logits = model(x)  # прямой проход
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))  # считаем loss
        optimizer.zero_grad()  # обнуляем градиенты предыдущего шага
        loss.backward()  # обратное распространение ошибки - считаем градиенты
        optimizer.step()  # AdamW обновляет параметры
        running += loss.item()  # накапливаем результаты расчета loss для вычисления среднего
        running_n += 1

        if step % log_every == 0 or step == steps:  # выводим логи каждые log_every итераций
            avg_train = running / running_n  # средний train_loss
            running, running_n = 0.0, 0  # сбрасываем значения
            val_l = eval_loss(model, val_ids, val_starts, context)  # средний val_loss по 20 фиксированным окнам
            curve.append((step, avg_train, val_l))  # запоминаем значения для train_curve.csv
            print(f"  step {step:5d}  train {avg_train:.4f}  val {val_l:.4f}")
        if step in sample_steps and step != 0:
            do_sample(step)  # генерируем образцы генерации предложений

    train_minutes = (time.time() - t0) / 60.0  # фиксируем время обучения в минутах

    # ---- запись конфига и кривой ----
    config = {
        "vocab_size": vocab_size,
        "d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
        "context": context, "steps": steps, "batch": batch,
        "lr": lr, "seed": SEED, "n_params": n_params,
        "train_tokens": int(train_ids.numel()),
        "val_tokens": int(val_ids.numel()),
        "train_minutes": round(train_minutes, 2),
        "device": device,
        "bpe_merges": len(merges),
    }
    (ARTIFACTS / "tiny_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(ARTIFACTS / "train_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        for row in curve:
            w.writerow(row)

    (ARTIFACTS / "samples.json").write_text(json.dumps({
        "prompt": prompt_text, "temperature": 0.8, "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- четыре режима генерации ----
    runs = []
    gen_specs = [
        ("greedy", 0.0, None),  # жадный режим
        ("t0.7", 0.7, None),
        ("t1.5", 1.5, None),
        ("top_p0.9", 1.0, 0.9),
    ]
    for name, temp, top_p in gen_specs:
        new_ids = generate_ids(model, prompt_ids, n_gen, temperature=temp,
                               top_p=top_p, context=context, gen=gen)
        text = detokenize(new_ids, id_to_tok)
        loop_start = find_loop_start(new_ids) if name == "greedy" else None  # ищем повторяющиеся токены
        runs.append({
            "name": name, "temperature": temp, "top_p": top_p,
            "text": text,
            "ids": new_ids, "n_tokens": len(new_ids),
            "loop_start": loop_start,
        })
    (ARTIFACTS / "generation.json").write_text(json.dumps({
        "model": "tiny", "prompt": prompt_text, "max_new_tokens": n_gen,
        "runs": runs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- сравнение с биграммами ----
    train_words = all_words[:split_word]
    val_words = all_words[split_word:split_word + eval_words]

    # биграммы: nats по словам
    bg = BigramLM(alpha=0.01)
    bg.fit(train_words)
    nats_bg = 0.0
    for v, w in zip(val_words, val_words[1:]):
        nats_bg += -bg.logprob(v, w)  # сумма −log P(w|v) по всем парам
    n_words_bg = len(val_words) - 1

    # трансформер: nats по токенам, затем делим на число слов
    val_tok_ids = []
    for w in val_words:
        for t in apply_bpe(w, merges):
            val_tok_ids.append(vocab.get(t, 0))
    nats_tiny = 0.0
    n_tok_tiny = 0
    with torch.no_grad():
        for i in range(0, len(val_tok_ids) - 1, context):
            chunk = val_tok_ids[i:i + context + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            logits = model(x)
            nats_tiny += F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), reduction="sum"
            ).item()  # сумма −log P по всем токенам
            n_tok_tiny += y.numel()  # сколько токенов оценили
    nats_per_word_tiny = nats_tiny / max(1, len(val_words))
    nats_per_word_bg = nats_bg / max(1, len(val_words))

    eval_out = {
        "words": len(val_words),
        "bigram": {
            "tokens": n_words_bg,
            "nats_total": nats_bg,
            "nats_per_word": nats_per_word_bg,
            "ppl_per_word": math.exp(nats_per_word_bg),
            "alpha": 0.01,
        },
        "tiny": {
            "tokens": n_tok_tiny,
            "nats_total": nats_tiny,
            "nats_per_word": nats_per_word_tiny,
            "ppl_per_word": math.exp(nats_per_word_tiny),
        },
    }
    (ARTIFACTS / "eval.json").write_text(
        json.dumps(eval_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Готово за {train_minutes:.1f} мин. bigram {nats_per_word_bg:.3f}, "
          f"tiny {nats_per_word_tiny:.3f} нат/слово")


if __name__ == "__main__":
    main()
