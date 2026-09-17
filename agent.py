"""DZ_6: Пайплайн тикета поддержки на графе состояний + контрольный слой.

Базовый сценарий (порт DZ_5) — 12 состояний:
    check_memory → classify → retrieve → check_relevance → generate
    → validate → check_validation → save → finish

плюс ветви:
    check_memory: hit ────────────────→ finish_cached
    classify: risk=high / parse_error → escalate
    check_relevance: нет контекста ────→ refuse
    check_validation: не grounded ────→ generate (ретрай) / escalate

Контрольный слой (новое в DZ_6):
    - состояние check_answer (13-е, между generate и validate) —
      детерминированная проверка ответа без LLM (пустота, длина, источники);
    - RunMetrics — метрики прогона: успех, длительность, LLM-вызовы, токены;
    - MeteredChat — метрирующая обёртка LLM: токены (usage или оценка len//4),
      retry с backoff на сетевые ошибки;
    - BudgetGuard — предохранитель MAX_LLM_CALLS_PER_RUN: превышение →
      эскалация budget_exceeded, а не падение;
    - RunLog — лог выполнения logs/runs.jsonl (JSONL, одна строка на прогон),
      сводная таблица метрик в --demo и агрегаты в --report;
    - алерт считает только *технические* эскалации: safety-эскалация
      high_risk — штатное безопасное поведение, не ошибка;
    - gold set (context/eval.json) + --eval — оценка качества ответов:
      ожидаемые факты и источник на вопрос, а не только успешный outcome.

Контекст — векторный поиск по базе знаний в Qdrant (Docker, паттерн DZ_4);
память — граф прошлых Q&A (memory/qa.json).
LLM — OpenAI-совместимый сервер (LM Studio); для selftest — заглушка FakeLLM
(режимы default/risky/ungrounded/loop/flaky/badformat) и мок Qdrant с
TF-IDF-эмбеддером, поэтому самодиагностика работает без LLM, без Qdrant
и без сети. Если Qdrant или LLM недоступны в живом режиме — агент не падает:
поиск деградирует до мок-памяти (детерминированные псевдо-векторы),
на LLM — эскалация с причиной, а не traceback.

Запуск:
    .venv/bin/python agent.py                 # интерактив
    .venv/bin/python agent.py "вопрос"         # один вопрос (+ строка метрик)
    .venv/bin/python agent.py --demo           # 4 прогона + сводная таблица
    .venv/bin/python agent.py --selftest       # 14 проверок без LLM
    .venv/bin/python agent.py --eval           # gold set: качество ответов
    .venv/bin/python agent.py --report         # агрегаты по logs/runs.jsonl
    .venv/bin/python agent.py --show-trace "вопрос"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Callable, Optional

import openai
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Конфигурация — читается из .env / переменных окружения
# --------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "lm-studio"
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-12b-qat")
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))

KB_FILE = os.getenv("KB_FILE", "context/kb.json")
QA_MEMORY_FILE = os.getenv("QA_MEMORY_FILE", "memory/qa.json")
# Gold set для оценки качества ответов (--eval): вопросы + ожидаемые факты.
EVAL_FILE = os.getenv("EVAL_FILE", "context/eval.json")

# Векторный поиск (Qdrant, паттерн DZ_4)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dz6_memory")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Порог релевантности — косинусное сходство (0..1) найденного контекста.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))
MEMORY_HIT_THRESHOLD = float(os.getenv("MEMORY_HIT_THRESHOLD", "1.5"))
MAX_VALIDATION_RETRIES = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))

# --- Контрольный слой (DZ_6) ------------------------------------------------ #

# Предохранитель: максимум LLM-вызовов за один прогон (каждая попытка —
# включая повторные после сетевой ошибки — учитывается в бюджете).
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "6"))
# Retry с backoff (0.5с/1.5с) только на транзиентные сетевые ошибки LLM.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Лимит длины ответа для детерминированной проверки check_answer.
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "2000"))
# Лог выполнения (JSONL, одна строка на прогон) — часть сдачи.
RUNS_LOG_FILE = os.getenv("RUNS_LOG_FILE", "logs/runs.jsonl")
# Максимум записей в логе: при переполнении файл ротируется в <файл>.1
# (хранится один архив). 0 — ротация выключена.
RUNS_LOG_MAX_RECORDS = int(os.getenv("RUNS_LOG_MAX_RECORDS", "1000"))
# Тариф за 1 000 000 токенов в рублях — для метрики cost_rub (локальной
# модели это оценочный тариф облачного эквивалента). Входные (prompt) и
# выходные (completion) токены тарифицируются отдельно.
LLM_COST_INPUT_PER_1M_RUB = float(os.getenv("LLM_COST_INPUT_PER_1M_RUB", "15"))
LLM_COST_OUTPUT_PER_1M_RUB = float(os.getenv("LLM_COST_OUTPUT_PER_1M_RUB", "60"))
# Порог доли эскалаций (0..1): выше — алерт (по умолчанию 0.2 — более 20% прогонов).
ERROR_ALERT_THRESHOLD = float(os.getenv("ERROR_ALERT_THRESHOLD", "0.2"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR")
logger = logging.getLogger("dz6.agent")

END = "END"  # маркер: сценарий завершён

# --- Классификация эскалаций --------------------------------------------------
# Safety-эскалации — штатное безопасное поведение агента (ветка risk=high),
# не ошибка: в алерт и «технические ошибки» не входят. Всё остальное —
# техническая ошибка (сбой состояния, бюджет, гард, провал валидаций).

SAFETY_ESCALATION_REASONS = frozenset({"high_risk"})


def is_safety_escalation(reason: Optional[str]) -> bool:
    """Штатная safety/business-эскалация (high_risk) — не ошибка."""
    return reason in SAFETY_ESCALATION_REASONS


def is_technical_escalation(reason: Optional[str]) -> bool:
    """Техническая ошибка (step_error:*, budget_exceeded, max_steps,
    classification_failed, grounding_validation_failed, answer_check_failed) —
    требует внимания."""
    return reason is not None and not is_safety_escalation(reason)


def _resolve(path: str) -> str:
    """Относительный путь — от корня проекта."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# --------------------------------------------------------------------------- #
# Модели данных
# --------------------------------------------------------------------------- #

class Outcome(StrEnum):
    ANSWERED = auto()
    ANSWERED_CACHED = auto()
    REFUSED = auto()
    ESCALATED = auto()


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    document: Document
    score: float


@dataclass(frozen=True)
class Node:
    """Узел графа: документ базы знаний (doc) или обработанный вопрос (qa)."""
    id: str
    label: str
    type: str  # "doc" | "qa"
    text: str


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    relation: str


@dataclass(frozen=True)
class ChatResponse:
    """Ответ LLM: текст + usage (None, если сервер usage не возвращает)."""
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class AgentResult:
    outcome: Outcome
    message: str
    sources: list[str] = field(default_factory=list)
    escalated_reason: str | None = None
    memory_saved: bool = False
    trace: list[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    """Метрики одного прогона: успех, длительность, стоимость, ретраи.

    Стоимость — в рублях (cost_rub): входные и выходные токены тарифицируются
    отдельно по тарифам за 1M токенов (LLM_COST_*_PER_1M_RUB).
    """
    duration_s: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    outcome: Optional[Outcome] = None
    budget_violated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_rub(self) -> float:
        """Стоимость прогона в рублях: входные и выходные токены по отдельным
        тарифам за 1 000 000 токенов."""
        return round(
            self.prompt_tokens * LLM_COST_INPUT_PER_1M_RUB / 1_000_000
            + self.completion_tokens * LLM_COST_OUTPUT_PER_1M_RUB / 1_000_000,
            6,
        )

    @property
    def success(self) -> bool:
        """ANSWERED / ANSWERED_CACHED — успех; REFUSED — нормальный отказ
        (считается отдельно); ESCALATED — ошибка."""
        return self.outcome in (Outcome.ANSWERED, Outcome.ANSWERED_CACHED)


@dataclass(frozen=True)
class ScenarioConfig:
    relevance_threshold: float = RELEVANCE_THRESHOLD
    memory_hit_threshold: float = MEMORY_HIT_THRESHOLD
    max_validation_retries: int = MAX_VALIDATION_RETRIES
    max_steps: int = MAX_STEPS
    max_answer_chars: int = MAX_ANSWER_CHARS


@dataclass
class WorkflowContext:
    """Состояние сценария, которое переживается между состояниями графа."""
    query: str
    classification: dict[str, Any] = field(default_factory=dict)
    docs: list[RetrievedDoc] = field(default_factory=list)
    answer: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    # Кто записал текущий провал: "check_answer" (детерминированная проверка)
    # или "" (LLM-валидатор) — определяет причину финальной эскалации.
    validation_source: str = ""
    retry_count: int = 0
    validation_ok: bool = False
    can_retry: bool = False
    relevance_ok: bool = False
    memory_hit: Optional[Node] = None
    memory_saved: bool = False
    escalate_reason: Optional[str] = None
    result: Optional[AgentResult] = None
    trace: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Keyword-скоринг (port из DZ_1/DZ_3)
# --------------------------------------------------------------------------- #

_STOP_WORDS = frozenset({
    "и", "в", "во", "на", "по", "с", "со", "из", "за", "от", "о", "об", "а", "но",
    "или", "же", "бы", "не", "да", "что", "как", "кто", "где", "когда", "почему",
    "зачем", "куда", "откуда", "можно", "нужно", "надо", "для", "до", "у", "то",
    "так", "какой", "какая", "какое", "какие", "это", "этот", "эта", "эти",
    "такой", "такая", "такое", "такие", "про", "через",
})


def keyword_score(query: str, text: str) -> float:
    """Скор keyword-совпадений запроса с текстом.

    Точное совпадение = 2, префикс (первые 4 символа) = 1, подстрока = 1 —
    префикс/подстрока только для токенов >= 4 с обеих сторон (иначе «с», «по»
    дают ложные хиты). Итог: сумма / число токенов запроса.
    """
    query_words = [w for w in query.lower().split() if len(w) > 2 and w not in _STOP_WORDS]
    if not query_words:
        query_words = query.lower().split()
    text_words = text.lower().split()
    score = 0
    for q in query_words:
        for t in text_words:
            if q == t:
                score += 2
                break
            if len(q) >= 4 and len(t) >= 4 and q[:4] == t[:4]:
                score += 1
                break
            if len(q) >= 4 and len(t) >= 4 and q in t:
                score += 1
                break
    return score / max(len(query_words), 1)


# --------------------------------------------------------------------------- #
# Граф-память (port из DZ_4)
# --------------------------------------------------------------------------- #

class GraphMemory:
    """Хранит узлы и рёбра (adjacency list, обе направленности)."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.outgoing.setdefault(node.id, [])
        self.incoming.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        self.outgoing.setdefault(edge.from_id, []).append(edge)
        self.incoming.setdefault(edge.to_id, []).append(edge)


def load_documents(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(d["doc_id"], d["title"], d["text"]) for d in data["documents"]]


def build_runtime_memory(kb: list[Document], qa_path: str) -> GraphMemory:
    """Граф рантайма: документы БЗ (kb.json) + прошлые Q&A (qa.json).

    Документы перечитываются при каждом старте, поэтому в файл памяти
    сохраняются только qa-узлы (см. save_qa_graph).
    """
    graph = GraphMemory()
    for doc in kb:
        graph.add_node(Node(doc.doc_id, doc.title, "doc", doc.text))
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for nd in data.get("nodes", []):
            graph.add_node(Node(nd["id"], nd["label"], nd["type"], nd["text"]))
        for eg in data.get("edges", []):
            graph.add_edge(Edge(eg["from"], eg["to"], eg["relation"]))
    return graph


def _next_qa_id(graph: GraphMemory) -> str:
    max_n = 0
    for nid in graph.nodes:
        if nid.startswith("qa-") and nid[3:].isdigit():
            max_n = max(max_n, int(nid[3:]))
    return f"qa-{max_n + 1}"


def save_qa_graph(graph: GraphMemory, path: str) -> None:
    """Сохраняет граф в файл: qa-узлы + все рёбра. Атомарно через .tmp."""
    nodes = [
        {"id": n.id, "label": n.label, "type": n.type, "text": n.text}
        for n in graph.nodes.values()
        if n.type == "qa"
    ]
    edges = [
        {"from": e.from_id, "to": e.to_id, "relation": e.relation}
        for edges_from in graph.outgoing.values()
        for e in edges_from
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Векторная память: эмбеддинги + Qdrant (port из DZ_4)
# --------------------------------------------------------------------------- #

def embed_with_llm(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Получает векторы эмбеддингов через LLM-сервер."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [[float(v) for v in d.embedding] for d in resp.data]


def _random_embed(text: str, dim: int) -> list[float]:
    """Детерминированный псевдо-вектор (хэш текста → RNG) — фолбэк без LLM."""
    import random as _r
    h = sum(ord(c) * (i + 1) for i, c in enumerate(text)) & 0xFFFFFFFF
    rng = _r.Random(h)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    """Токенизация: нижний регистр, только слова, без стоп-слов, len >= 2."""
    tokens = re.split(r"[^\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) >= 2]


class TfidfEmbedder:
    """Детерминированный TF-IDF «эмбеддер» для selftest (без сети).

    Строит IDF по корпусу, затем выдаёт TF-IDF векторы по общему словарю —
    в отличие от псевдо-векторов, поиск на нём остаётся осмысленным.
    """

    def __init__(self, corpus: list[str], dim: int = 512):
        self.dim = dim
        n_docs = len(corpus) or 1
        doc_freq: Counter = Counter()
        for text in corpus:
            for t in set(_tokenize(text)):
                doc_freq[t] += 1
        self.vocabulary = [w for w, _ in doc_freq.most_common(dim)]
        self.idf: dict[str, float] = {}
        for w in self.vocabulary:
            df = doc_freq.get(w, 0) or 1
            self.idf[w] = math.log((n_docs + 1) / (df + 1)) + 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            freq: Counter = Counter(_tokenize(text))
            total = len(freq) or 1
            vec = [
                (freq.get(w, 0) / total) * self.idf.get(w, 1.0)
                for w in self.vocabulary
            ]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class QdrantMemory:
    """Векторная память на Qdrant: upsert документов + cosine-поиск."""

    def __init__(self, url: str, collection: str,
                 embed_fn: Callable[[list[str]], list[list[float]]]):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embed_fn = embed_fn
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию (cosine), если её ещё нет."""
        from qdrant_client.http.models import Distance, VectorParams
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert_documents(self, docs: list[Document]) -> None:
        texts = [f"{d.title} {d.text}" for d in docs]
        vectors = self.embed_fn(texts)
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"doc_id": d.doc_id, "title": d.title, "text": d.text},
            )
            for i, (d, vec) in enumerate(zip(docs, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        """Возвращает [(doc_id, cosine_score, payload), ...]."""
        vec = self.embed_fn([query])[0]
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
        )
        return [(p.payload["doc_id"], p.score, p.payload) for p in resp.points]

    def close(self) -> None:
        self.client.close()


class MockQdrantMemory:
    """Заглушка Qdrant (selftest и фолбэк): тот же интерфейс, косинус в памяти."""

    def __init__(self, embed_fn: Callable[[list[str]], list[list[float]]]):
        self.embed_fn = embed_fn
        self.points: list[dict] = []

    def upsert_documents(self, docs: list[Document]) -> None:
        texts = [f"{d.title} {d.text}" for d in docs]
        for d, vec in zip(docs, self.embed_fn(texts)):
            self.points.append({
                "doc_id": d.doc_id,
                "vector": vec,
                "payload": {"doc_id": d.doc_id, "title": d.title, "text": d.text},
            })

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        qvec = self.embed_fn([query])[0]
        scored = [
            (p["doc_id"], cosine_similarity(qvec, p["vector"]), p["payload"])
            for p in self.points
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def close(self) -> None:
        pass


def make_vector_memory(client: openai.OpenAI, kb: list[Document]):
    """Векторная память: Qdrant, если доступен, иначе мок. Никогда не падает.

    LLM (embedding-модель) недоступен → детерминированные псевдо-векторы
    (поиск деградирует до «почти нет сходства» → ветка refuse, не crash).
    Qdrant недоступен → MockQdrantMemory с теми же эмбеддингами.
    """
    embed_fn: Callable[[list[str]], list[list[float]]]
    try:
        embed_fn = lambda texts: embed_with_llm(client, texts)
        embed_fn(["selftest"])  # проверка, что сервер отдаёт эмбеддинги
        print(f"Эмбеддинги: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    except Exception as e:
        print(f"[ошибка LLM] эмбеддинги недоступны ({e.__class__.__name__}). "
              f"Использую детерминированные псевдо-векторы.")
        embed_fn = lambda texts: [_random_embed(t, EMBEDDING_DIM) for t in texts]
    try:
        qmem = QdrantMemory(QDRANT_URL, QDRANT_COLLECTION, embed_fn)
        qmem.upsert_documents(kb)
        print(f"Qdrant: {QDRANT_COLLECTION} загружено ({len(kb)} документов)")
    except Exception as e:
        print(f"[ошибка Qdrant] {e.__class__.__name__}: {str(e)[:200]}. Использую мок.")
        qmem = MockQdrantMemory(embed_fn)
        qmem.upsert_documents(kb)
    return qmem


# --------------------------------------------------------------------------- #
# Контрольный слой: метрики, бюджет, retry
# --------------------------------------------------------------------------- #

RawChatFn = Callable[[str, str], ChatResponse]  # (system, user) -> ответ
ChatFn = Callable[[str, str], str]  # (system, user) -> текст (после обёртки)


def _usage_tokens(resp: ChatResponse, system: str, user: str) -> tuple[int, int]:
    """Токены: из usage, если сервер его отдаёт; иначе оценка len(text)//4.

    Одна и та же формула для реального клиента и FakeLLM — детерминизм selftest.
    """
    prompt = resp.prompt_tokens
    completion = resp.completion_tokens
    if prompt is None:
        prompt = (len(system) + len(user)) // 4
    if completion is None:
        completion = len(resp.text) // 4
    return prompt, completion


class BudgetExceeded(Exception):
    """Бюджет LLM-вызовов за прогон исчерпан (предохранитель сработал)."""


class BudgetGuard:
    """Предохранитель: максимум LLM-вызовов за один прогон.

    Перед каждым вызовом (каждая попытка, включая повторные после сетевой
    ошибки) проверяется llm_calls < max_calls; превышение — BudgetExceeded,
    который движок переводит в эскалацию, а не падение.
    """

    def __init__(self, max_calls: int = MAX_LLM_CALLS_PER_RUN) -> None:
        self.max_calls = max_calls
        self.metrics: RunMetrics = RunMetrics()

    def before_call(self) -> None:
        if self.metrics.llm_calls >= self.max_calls:
            raise BudgetExceeded(
                f"бюджет LLM-вызовов исчерпан: {self.metrics.llm_calls} >= {self.max_calls}"
            )


class MeteredChat:
    """Метрирующая обёртка над LLM: счётчик вызовов, токены, retry, бюджет.

    - каждый вызов (попытка) учитывается в бюджете и метриках прогона;
    - токены — из resp.usage, фолбэк len(text)//4 (см. _usage_tokens);
    - retry с backoff (0.5с, 1.5с) только на транзиентные сетевые ошибки
      (APIConnectionError/APITimeoutError), до max_retries; 4xx и прочие
      ошибки не ретраются — пробрасываются вызывающему.
    """

    def __init__(self, raw: RawChatFn, max_retries: int = LLM_MAX_RETRIES) -> None:
        self._raw = raw
        self.max_retries = max_retries
        self.backoff_s = (0.5, 1.5)
        self.guard = BudgetGuard()

    def new_run(self) -> RunMetrics:
        """Новый прогон: свежие метрики и сброс счётчика бюджета."""
        self.guard.metrics = RunMetrics()
        return self.guard.metrics

    def __call__(self, system: str, user: str) -> str:
        m = self.guard.metrics
        for attempt in range(self.max_retries + 1):
            self.guard.before_call()
            m.llm_calls += 1
            try:
                resp = self._raw(system, user)
            except (openai.APIConnectionError, openai.APITimeoutError):
                # Только транзиентные сетевые ошибки ретраем; остальные — пробрасываем.
                if attempt >= self.max_retries:
                    raise
                m.retries += 1
                time.sleep(self.backoff_s[min(attempt, len(self.backoff_s) - 1)])
                continue
            p, c = _usage_tokens(resp, system, user)
            m.prompt_tokens += p
            m.completion_tokens += c
            return resp.text
        raise AssertionError("недостижимо: цикл retry завершился без ответа")


# --------------------------------------------------------------------------- #
# LLM-вызовы
# --------------------------------------------------------------------------- #

def make_openai_chat(client: openai.OpenAI) -> RawChatFn:
    """Реальный LLM: OpenAI-совместимый сервер (LM Studio)."""

    def raw_chat(system: str, user: str) -> ChatResponse:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return ChatResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    return raw_chat


def _chat_json(chat: ChatFn, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Вызывает модель и парсит JSON-ответ; при сбое парсинга не роняем пайплайн."""
    raw = chat(system_prompt, user_prompt).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Модель не всегда строго следует формату — деградируем в безопасную
        # сторону (ветку выбирает маршрутизация вызывающего состояния).
        return {"_parse_error": True, "_raw": raw}


def _fake_network_error() -> openai.APIConnectionError:
    """Имитация сетевой ошибки LLM (selftest: режим flaky)."""
    import httpx
    return openai.APIConnectionError(request=httpx.Request("POST", "http://fake.local/v1/chat"))


class FakeLLM:
    """Детерминированная LLM-заглушка для selftest (без сети).

    Режимы:
      default    — низкий риск, валидатор принимает ответ;
      risky      — классификатор возвращает risk="high";
      ungrounded — валидатор всегда отвергает ответ;
      loop       — как ungrounded (для проверки гарда по шагам);
      flaky      — первые 2 вызова «падают» сетевой ошибкой (проверка retry);
      badformat  — генерация ссылается на несуществующий doc-fake
                   (проверка check_answer).
    """

    def __init__(self, mode: str = "default") -> None:
        self.mode = mode
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> ChatResponse:
        self.calls.append((system, user))
        if self.mode == "flaky" and len(self.calls) <= 2:
            raise _fake_network_error()
        if "классификатор" in system:
            risk = "high" if self.mode == "risky" else "low"
            return ChatResponse(json.dumps({
                "category": "complaint" if risk == "high" else "faq",
                "risk": risk,
                "complexity": "simple",
                "language": "ru",
            }, ensure_ascii=False))
        if "валидатор" in system:
            grounded = self.mode not in ("ungrounded", "loop")
            return ChatResponse(json.dumps({
                "grounded": grounded,
                "reason": "все утверждения подтверждены контекстом"
                if grounded else "в ответе есть факты, которых нет в контексте",
            }, ensure_ascii=False))
        # Генерация: doc-id берём из маркеров [doc_id] в контекстном блоке.
        if self.mode == "badformat":
            return ChatResponse("Ответ составлен по базе знаний. [источники: doc-fake]")
        doc_ids = list(dict.fromkeys(re.findall(r"\[([a-z0-9][a-z0-9-]*)\]", user)))
        src = ", ".join(doc_ids) if doc_ids else "документы не указаны"
        return ChatResponse(f"Ответ составлен по базе знаний. [источники: {src}]")


# --------------------------------------------------------------------------- #
# Промпты (port из DZ_1)
# --------------------------------------------------------------------------- #

CLASSIFY_SYSTEM_PROMPT = """\
Ты — классификатор входящих запросов в поддержку.
Твоя задача: проанализировать запрос пользователя и определить его категорию, уровень риска, сложность и язык.

Сначала кратко проанализируй запрос, а затем верни ТОЛЬКО JSON без пояснений и без markdown-разметки в формате:
{
  "category": "faq" | "complaint" | "other",
  "risk": "low" | "high",
  "complexity": "simple" | "complex",
  "language": "ru" | "en" | "other"
}
risk="high" — если запрос содержит жалобу, угрозу, юридическую тему,
упоминание вреда себе или другим, оскорбления.
complexity="complex" — если вопрос требует синтеза нескольких фактов
или неоднозначен.
"""

ANSWER_SYSTEM_PROMPT = """\
Ты — агент поддержки. Отвечай ТОЛЬКО на основании предоставленного контекста
из базы знаний. Если контекста недостаточно — явно скажи об этом, не
придумывай факты. В конце ответа перечисли id использованных документов
в формате: [источники: doc_id1, doc_id2].
"""

VALIDATE_SYSTEM_PROMPT = """\
Ты — валидатор ответов поддержки. Тебе дан контекст и сгенерированный ответ.
Твоя задача: проверить, что КАЖДОЕ фактическое утверждение в ответе подтверждается контекстом.

Сначала проведи тщательный сравнительный анализ фактов из контекста и ответа.
Затем верни ТОЛЬКО JSON:
{ "grounded": true | false, "reason": "краткое объяснение" }
"""


# --------------------------------------------------------------------------- #
# Движок графа состояний
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class State:
    """Состояние графа: действие (модифицирует контекст) и функция маршрутизации."""
    name: str
    action: Callable[[WorkflowContext], None]
    route: Callable[[WorkflowContext], str]


class Workflow:
    """Движок исполнения графа состояний с гардом по числу шагов и бюджету."""

    def __init__(self, states: dict[str, State], entry: str, max_steps: int) -> None:
        self.states = states
        self.entry = entry
        self.max_steps = max_steps

    def run(self, ctx: WorkflowContext) -> AgentResult:
        current = self.entry
        while current != END:
            # Гард от зацикливания: лимит на число состояний за прогон
            # (петля generate → check_answer → validate → check_validation
            # — реальный цикл графа).
            if len(ctx.trace) >= self.max_steps:
                ctx.result = AgentResult(
                    outcome=Outcome.ESCALATED,
                    message="Сценарий не завершился за отведённое число шагов, передаю оператору.",
                    escalated_reason="max_steps",
                    trace=ctx.trace,
                )
                return ctx.result
            state = self.states[current]
            ctx.trace.append(state.name)
            try:
                state.action(ctx)
            except BudgetExceeded as e:
                # Предохранитель по бюджету LLM-вызовов: эскалация, а не падение.
                ctx.result = AgentResult(
                    outcome=Outcome.ESCALATED,
                    message="Исчерпан бюджет LLM-вызовов за прогон, передаю оператору.",
                    escalated_reason="budget_exceeded",
                    trace=ctx.trace,
                )
                logger.info("бюджет исчерпан в состоянии %s: %s", state.name, e)
                return ctx.result
            except Exception as e:
                # Ошибка в состоянии — эскалация, а не падение всего прогона.
                ctx.result = AgentResult(
                    outcome=Outcome.ESCALATED,
                    message="При обработке запроса произошла ошибка, передаю оператору.",
                    escalated_reason=f"step_error:{state.name}",
                    trace=ctx.trace,
                )
                logger.exception("состояние %s упало: %s", state.name, e)
                return ctx.result
            current = state.route(ctx)
            if current != END and current not in self.states:
                # Защита от «фантомного» перехода — только если граф собран с ошибкой.
                ctx.result = AgentResult(
                    outcome=Outcome.ESCALATED,
                    message="При обработке запроса произошла ошибка, передаю оператору.",
                    escalated_reason=f"unknown_state:{current}",
                    trace=ctx.trace,
                )
                return ctx.result
        assert ctx.result is not None, "терминальное состояние не установило результат"
        return ctx.result


# --------------------------------------------------------------------------- #
# Сценарий: тикет поддержки (5 шагов + 5 ветвлений, 13 состояний)
# --------------------------------------------------------------------------- #

def _answer_problem(answer: str, known_ids: set[str], retrieved_ids: set[str],
                    max_chars: int) -> Optional[str]:
    """Детерминированная проверка ответа. Возвращает причину провала или None.

    - ответ непустой;
    - длина <= max_chars;
    - id в блоке [источники: …] существуют в базе знаний
      (защита от «галлюцинированных» источников);
    - id в блоке [источники: …] — среди документов, реально полученных
      текущим поиском (top-k): модель может цитировать только то,
      что увидела в контексте.
    """
    if not answer.strip():
        return "ответ пуст"
    if len(answer) > max_chars:
        return f"ответ длиннее {max_chars} символов (сейчас {len(answer)})"
    m = re.search(r"\[источники:\s*([^\]]+)\]", answer)
    if m:
        cited = [t.strip() for t in m.group(1).split(",")]
        cited = [t for t in cited if t and t != "документы не указаны"]
        unknown = [t for t in cited if t not in known_ids]
        if unknown:
            return f"в источниках несуществующие id: {', '.join(unknown)}"
        stale = [t for t in cited if t not in retrieved_ids]
        if stale:
            return f"источники не получены текущим поиском: {', '.join(stale)}"
    return None


def build_scenario(
    chat: ChatFn,
    kb: list[Document],
    memory: GraphMemory,
    qmem,
    qa_path: str,
    config: ScenarioConfig,
) -> Workflow:
    """Собирает граф состояний сценария. Зависимости захватываются замыканиями."""
    known_doc_ids = {d.doc_id for d in kb}

    # -- check_memory: вопрос уже был? ---------------------------------------
    def _act_check_memory(ctx: WorkflowContext) -> None:
        best_node: Optional[Node] = None
        best_score = 0.0
        for node in memory.nodes.values():
            if node.type != "qa":
                continue
            s = keyword_score(ctx.query, node.label)
            if s > best_score:
                best_node, best_score = node, s
        if best_node is not None and best_score >= config.memory_hit_threshold:
            ctx.memory_hit = best_node

    def _route_check_memory(ctx: WorkflowContext) -> str:
        return "finish_cached" if ctx.memory_hit is not None else "classify"

    # -- classify: классификация LLM (шаг 1) ----------------------------------
    def _act_classify(ctx: WorkflowContext) -> None:
        ctx.classification = _chat_json(chat, CLASSIFY_SYSTEM_PROMPT, ctx.query)

    def _route_classify(ctx: WorkflowContext) -> str:
        c = ctx.classification
        if c.get("_parse_error"):
            ctx.escalate_reason = "classification_failed"
            return "escalate"
        if c.get("risk") == "high":
            ctx.escalate_reason = "high_risk"
            return "escalate"
        return "retrieve"

    # -- retrieve: векторный поиск по базе знаний (шаг 2) ----------------------
    def _act_retrieve(ctx: WorkflowContext) -> None:
        docs_by_id = {d.doc_id: d for d in kb}
        results = qmem.search(ctx.query, top_k=3)
        # Косинусное сходство может быть отрицательным — ограничиваем скор
        # до [0, 1]: пороги и выводы работают в фиксированном диапазоне.
        ctx.docs = [
            RetrievedDoc(document=docs_by_id[doc_id],
                         score=round(min(1.0, max(0.0, score)), 3))
            for doc_id, score, _ in results
            if doc_id in docs_by_id
        ]

    def _route_retrieve(ctx: WorkflowContext) -> str:
        return "check_relevance"

    # -- check_relevance: проверка «контекста достаточно?» ----------------------
    def _act_check_relevance(ctx: WorkflowContext) -> None:
        best = max((d.score for d in ctx.docs), default=0.0)
        ctx.relevance_ok = best >= config.relevance_threshold

    def _route_check_relevance(ctx: WorkflowContext) -> str:
        return "generate" if ctx.relevance_ok else "refuse"

    # -- generate: ответ LLM (шаг 3) --------------------------------------------
    def _act_generate(ctx: WorkflowContext) -> None:
        context_block = "\n\n".join(
            f"[{d.document.doc_id}] {d.document.title}\n{d.document.text}"
            for d in ctx.docs
        )
        user = f"Контекст:\n{context_block}\n\nВопрос пользователя:\n{ctx.query}"
        if ctx.retry_count > 0 and ctx.validation.get("reason"):
            user += (
                "\n\nПредыдущий ответ не прошёл валидацию. "
                f"Комментарий валидатора: {ctx.validation['reason']}"
            )
        ctx.answer = chat(ANSWER_SYSTEM_PROMPT, user)

    def _route_generate(ctx: WorkflowContext) -> str:
        return "check_answer"

    # -- check_answer: детерминированная проверка ответа (без LLM) --------------
    def _act_check_answer(ctx: WorkflowContext) -> None:
        # Источники сверяем и с БЗ (галлюцинации), и с топ-k текущего поиска —
        # модель видела в контексте только эти документы.
        retrieved_ids = {d.document.doc_id for d in ctx.docs}
        reason = _answer_problem(ctx.answer, known_doc_ids, retrieved_ids,
                                 config.max_answer_chars)
        if reason:
            # Провал оформляем как валидационный — ретраи работают как обычно,
            # а причина финальной эскалации будет answer_check_failed.
            ctx.validation = {"grounded": False, "reason": f"проверка ответа: {reason}"}
            ctx.validation_source = "check_answer"
        else:
            ctx.validation_source = ""

    def _route_check_answer(ctx: WorkflowContext) -> str:
        return "check_validation" if ctx.validation_source == "check_answer" else "validate"

    # -- validate: grounding-проверка (шаг 4) -----------------------------------
    def _act_validate(ctx: WorkflowContext) -> None:
        context_block = "\n\n".join(d.document.text for d in ctx.docs)
        user = f"Контекст:\n{context_block}\n\nОтвет для проверки:\n{ctx.answer}"
        ctx.validation = _chat_json(chat, VALIDATE_SYSTEM_PROMPT, user)
        ctx.validation_source = ""

    def _route_validate(ctx: WorkflowContext) -> str:
        return "check_validation"

    # -- check_validation: проверка «ответ подтверждён?» -------------------------
    def _act_check_validation(ctx: WorkflowContext) -> None:
        ctx.validation_ok = ctx.validation.get("grounded") is True
        if not ctx.validation_ok:
            # Ретрай возможен, пока не исчерпан лимит попыток.
            ctx.can_retry = ctx.retry_count < config.max_validation_retries
            if ctx.can_retry:
                ctx.retry_count += 1

    def _route_check_validation(ctx: WorkflowContext) -> str:
        if ctx.validation_ok:
            return "save"
        if ctx.can_retry:
            return "generate"
        # Причина зависит от того, кто отверг ответ: детерминированная
        # проверка (check_answer) или LLM-валидатор.
        ctx.escalate_reason = (
            "answer_check_failed" if ctx.validation_source == "check_answer"
            else "grounding_validation_failed"
        )
        return "escalate"

    # -- save: сохранение в память (шаг 5) ---------------------------------------
    def _act_save(ctx: WorkflowContext) -> None:
        node_id = _next_qa_id(memory)
        memory.add_node(Node(node_id, ctx.query, "qa", ctx.answer))
        for rd in ctx.docs:
            if rd.document.doc_id in memory.nodes:
                memory.add_edge(Edge(node_id, rd.document.doc_id, "использует"))
        save_qa_graph(memory, qa_path)
        ctx.memory_saved = True

    def _route_save(ctx: WorkflowContext) -> str:
        return "finish"

    # -- Терминальные состояния ---------------------------------------------------
    def _act_finish(ctx: WorkflowContext) -> None:
        ctx.result = AgentResult(
            outcome=Outcome.ANSWERED,
            message=ctx.answer,
            sources=[d.document.doc_id for d in ctx.docs],
            memory_saved=ctx.memory_saved,
            trace=ctx.trace,
        )

    def _act_refuse(ctx: WorkflowContext) -> None:
        ctx.result = AgentResult(
            outcome=Outcome.REFUSED,
            message="В базе знаний не нашлось релевантной информации. "
                    "Попробуйте переформулировать вопрос или обратитесь к ментору.",
            trace=ctx.trace,
        )

    def _act_escalate(ctx: WorkflowContext) -> None:
        ctx.result = AgentResult(
            outcome=Outcome.ESCALATED,
            message="Передаю ваш запрос оператору поддержки.",
            escalated_reason=ctx.escalate_reason or "unknown",
            trace=ctx.trace,
        )

    def _act_finish_cached(ctx: WorkflowContext) -> None:
        hit = ctx.memory_hit
        sources = [e.to_id for e in memory.outgoing.get(hit.id, []) if e.relation == "использует"]
        ctx.result = AgentResult(
            outcome=Outcome.ANSWERED_CACHED,
            message=f"(ответ из памяти) {hit.text}",
            sources=sources,
            trace=ctx.trace,
        )

    def _to_end(ctx: WorkflowContext) -> str:
        return END

    return Workflow({
        "check_memory": State("check_memory", _act_check_memory, _route_check_memory),
        "classify": State("classify", _act_classify, _route_classify),
        "retrieve": State("retrieve", _act_retrieve, _route_retrieve),
        "check_relevance": State("check_relevance", _act_check_relevance, _route_check_relevance),
        "generate": State("generate", _act_generate, _route_generate),
        "check_answer": State("check_answer", _act_check_answer, _route_check_answer),
        "validate": State("validate", _act_validate, _route_validate),
        "check_validation": State("check_validation", _act_check_validation, _route_check_validation),
        "save": State("save", _act_save, _route_save),
        "finish": State("finish", _act_finish, _to_end),
        "refuse": State("refuse", _act_refuse, _to_end),
        "escalate": State("escalate", _act_escalate, _to_end),
        "finish_cached": State("finish_cached", _act_finish_cached, _to_end),
    }, entry="check_memory", max_steps=config.max_steps)


def run_scenario(query: str, workflow: Workflow) -> AgentResult:
    """Один прогон сценария по вопросу."""
    return workflow.run(WorkflowContext(query=query))


def _execute(query: str, workflow: Workflow,
             meter: MeteredChat) -> tuple[AgentResult, RunMetrics]:
    """Один прогон с метриками: время, LLM-вызовы, токены, ретраи, outcome."""
    metrics = meter.new_run()
    start = time.monotonic()
    result = workflow.run(WorkflowContext(query=query))
    metrics.duration_s = round(time.monotonic() - start, 3)
    metrics.outcome = result.outcome
    if result.escalated_reason == "budget_exceeded":
        metrics.budget_violated = True
    return result, metrics


# --------------------------------------------------------------------------- #
# Лог выполнения (JSONL)
# --------------------------------------------------------------------------- #

class RunLog:
    """Лог выполнения: JSONL, одна JSON-строка на прогон, append-режим.

    Ротация: когда записей становится >= max_records, файл переименовывается
    в <файл>.1 (предыдущий архив не сохраняется), и лог начинается заново —
    runs.jsonl не растёт бесконечно. max_records <= 0 — ротация выключена.
    """

    def __init__(self, path: str, max_records: int = RUNS_LOG_MAX_RECORDS) -> None:
        self.path = path
        self.max_records = max_records

    def append(self, record: dict[str, Any]) -> None:
        self._rotate_if_needed()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self) -> None:
        if self.max_records <= 0 or not os.path.exists(self.path):
            return
        if len(self.read()) < self.max_records:
            return
        archive = self.path + ".1"
        if os.path.exists(archive):
            os.remove(archive)
        os.replace(self.path, archive)

    def read(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("некорректная строка в логе: %s", line[:100])
        return records


def log_run(run_log: RunLog, query: str, result: AgentResult,
            metrics: RunMetrics) -> dict[str, Any]:
    """Формирует запись о прогоне и дописывает её в лог выполнения."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "outcome": result.outcome.value,
        "escalated_reason": result.escalated_reason,
        "trace": result.trace,
        "duration_s": metrics.duration_s,
        "llm_calls": metrics.llm_calls,
        "tokens": {
            "prompt": metrics.prompt_tokens,
            "completion": metrics.completion_tokens,
            "total": metrics.total_tokens,
        },
        "cost_rub": metrics.cost_rub,
        "retries": metrics.retries,
        "memory_saved": result.memory_saved,
        "sources": result.sources,
    }
    run_log.append(record)
    return record


# --------------------------------------------------------------------------- #
# Оценка качества ответов: gold set (вопросы + ожидаемые факты)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalCase:
    """Gold-кейс: вопрос, ожидаемый итог, ожидаемые факты ответа и источник."""
    id: str
    query: str
    expect_outcome: str  # "answered" (включая answered_cached) | "refused" | "escalated"
    expect_facts: tuple[str, ...]
    expect_source: Optional[str]


@dataclass
class EvalCaseResult:
    """Результат прогона gold-кейса: итог, доля фактов, источник, стоимость."""
    case: EvalCase
    result: Optional[AgentResult]
    metrics: Optional[RunMetrics]
    facts_matched: int
    facts_total: int
    source_ok: bool
    passed: bool
    error: Optional[str] = None


def load_eval_cases(path: str) -> list[EvalCase]:
    """Загружает gold set и валидирует его (известный outcome, непустые вопросы)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases: list[EvalCase] = []
    for c in data.get("cases", []):
        outcome = c["expect_outcome"]
        if outcome not in ("answered", "refused", "escalated"):
            raise ValueError(f"кейс {c.get('id')}: неизвестный expect_outcome «{outcome}»")
        if not str(c.get("query", "")).strip():
            raise ValueError(f"кейс {c.get('id')}: пустой вопрос")
        cases.append(EvalCase(
            id=str(c["id"]),
            query=str(c["query"]),
            expect_outcome=outcome,
            expect_facts=tuple(c.get("expect_facts", [])),
            expect_source=c.get("expect_source"),
        ))
    if not cases:
        raise ValueError(f"gold set пуст: {path}")
    return cases


def _normalize(text: str) -> str:
    """Нормализация для сверки фактов: нижний регистр, схлопывание пробелов."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def match_facts(answer: str, facts: list[str]) -> tuple[list[str], list[str]]:
    """Какие ожидаемые факты (подстроки) есть в ответе. → (есть, отсутствуют)."""
    a = _normalize(answer)
    matched = [f for f in facts if _normalize(f) in a]
    missed = [f for f in facts if _normalize(f) not in a]
    return matched, missed


def _outcome_ok(expect: str, actual: Outcome) -> bool:
    """«answered» засчитывается и для answered, и для answered_cached."""
    if expect == "answered":
        return actual in (Outcome.ANSWERED, Outcome.ANSWERED_CACHED)
    return actual.value == expect


def run_eval(cases: list[EvalCase], workflow: Workflow,
             meter: MeteredChat) -> list[EvalCaseResult]:
    """Прогоняет gold set: по каждому кейсу — итог, факты, источник, метрики.

    Кейс PASSED: ожидаемый итог + все ожидаемые факты в ответе +
    ожидаемый источник в sources. Ошибка LLM — ERROR по кейсу, не падение.
    """
    results: list[EvalCaseResult] = []
    for case in cases:
        try:
            result, metrics = _execute(case.query, workflow, meter)
        except openai.APIError as e:
            results.append(EvalCaseResult(
                case=case, result=None, metrics=None,
                facts_matched=0, facts_total=len(case.expect_facts),
                source_ok=False, passed=False,
                error=f"{e.__class__.__name__}: {str(e)[:200]}",
            ))
            continue
        matched, _missed = match_facts(result.message, list(case.expect_facts))
        facts_total = len(case.expect_facts)
        source_ok = (case.expect_source in result.sources) if case.expect_source else True
        passed = (_outcome_ok(case.expect_outcome, result.outcome)
                  and len(matched) == facts_total and source_ok)
        results.append(EvalCaseResult(
            case=case, result=result, metrics=metrics,
            facts_matched=len(matched), facts_total=facts_total,
            source_ok=source_ok, passed=passed,
        ))
    return results


def print_eval_report(results: list[EvalCaseResult]) -> int:
    """Таблица по кейсам + сводка качества. Возвращает число не пройденных."""
    print(f"{'№':<4}{'кейс':<14}{'итог':<18}{'факты':>7}{'источник':>10}"
          f"{'стоимость':>12}  результат")
    for i, er in enumerate(results, 1):
        if er.result is None:
            print(f"{i:<4}{er.case.id:<14}{'—':<18}{'—':>7}{'—':>10}{'—':>12}  "
                  f"ERROR: {er.error}")
            continue
        facts = f"{er.facts_matched}/{er.facts_total}" if er.facts_total else "—"
        source = "ок" if er.source_ok else "нет"
        cost = f"{er.metrics.cost_rub:.4f}₽" if er.metrics else "—"
        print(f"{i:<4}{er.case.id:<14}{er.result.outcome.value:<18}{facts:>7}"
              f"{source:>10}{cost:>12}  {'PASS' if er.passed else 'FAIL'}")
        if not er.passed:
            if er.result.outcome.value != er.case.expect_outcome:
                print(f"       итог: {er.result.outcome.value} "
                      f"(ожидалось {er.case.expect_outcome})")
            _, missed = match_facts(er.result.message, list(er.case.expect_facts))
            if missed:
                print(f"       нет фактов: {', '.join(missed)}")
            if not er.source_ok:
                print(f"       нет источника: {er.case.expect_source} "
                      f"(фактически: {', '.join(er.result.sources) or '—'})")
    total = len(results)
    passed = sum(1 for er in results if er.passed)
    f_total = sum(er.facts_total for er in results)
    f_matched = sum(er.facts_matched for er in results)
    facts_str = (f"{f_matched}/{f_total} ({f_matched * 100 / f_total:.1f}%)"
                 if f_total else "—")
    print(f"Итог: кейсов пройдено {passed}/{total} ({passed * 100 / total:.1f}%) "
          f"| факты: {facts_str}")
    return total - passed


def run_eval_cli(kb: list[Document], qmem, raw_chat: RawChatFn) -> int:
    """--eval: gold set в изолированной среде (tmp-память, без записи в лог)."""
    cases = load_eval_cases(_resolve(EVAL_FILE))
    qa_dir = tempfile.mkdtemp(prefix="dz6-eval-")
    qa_path = os.path.join(qa_dir, "qa.json")
    # Свежая память: doc-узлы из БЗ, qa-узлов нет — каждый кейс идёт
    # полным пайплайном, не касаясь memory/qa.json и logs/runs.jsonl.
    memory = build_runtime_memory(kb, qa_path)
    meter = MeteredChat(raw_chat)
    workflow = build_scenario(meter, kb, memory, qmem, qa_path, ScenarioConfig())
    print(f"Оценка качества (gold set): {len(cases)} кейсов из "
          f"{os.path.relpath(_resolve(EVAL_FILE), BASE_DIR)}; память и лог прогонов не трогаются")
    results = run_eval(cases, workflow, meter)
    failed = print_eval_report(results)
    return 0 if failed == 0 else 1


# --------------------------------------------------------------------------- #
# Вывод и CLI
# --------------------------------------------------------------------------- #

def print_result(result: AgentResult, show_trace: bool = True) -> None:
    if show_trace and result.trace:
        print(f"Путь по состояниям: {' → '.join(result.trace)}")
    print(f"[{result.outcome.value}]")
    print(result.message)
    if result.sources:
        print(f"Источники: {', '.join(result.sources)}")
    if result.escalated_reason:
        print(f"Причина: {result.escalated_reason}")
    if result.memory_saved:
        print("Сохранено в память (qa.json).")


def print_metrics(metrics: RunMetrics) -> None:
    """Одна строка метрик прогона (для одиночного режима и интерактива)."""
    print(
        f"Метрики: {metrics.duration_s:.2f}с | LLM-вызовов: {metrics.llm_calls} | "
        f"токены: {metrics.total_tokens} (prompt={metrics.prompt_tokens}, "
        f"completion={metrics.completion_tokens}) | стоимость: {metrics.cost_rub:.4f}₽ "
        f"| ретраев: {metrics.retries}"
    )


def print_metrics_table(runs: list[tuple[str, AgentResult, RunMetrics]]) -> None:
    """Сводная таблица «прогон × метрики» + агрегаты + алерт (конец --demo)."""
    if not runs:
        return
    print("\n===== Сводка метрик =====")
    print(f"{'№':<4}{'результат':<18}{'LLM':>5}{'токены':>9}{'время':>10}{'ретраи':>9}"
          f"{'стоимость':>12}  комментарий")
    for i, (comment, result, m) in enumerate(runs, 1):
        note = result.escalated_reason or comment
        print(
            f"{i:<4}{result.outcome.value:<18}{m.llm_calls:>5}{m.total_tokens:>9}"
            f"{m.duration_s:>9.2f}с{m.retries:>9}{m.cost_rub:>11.4f}₽  {note}"
        )
    total = len(runs)
    ok = sum(1 for _, r, _ in runs if r.outcome in (Outcome.ANSWERED, Outcome.ANSWERED_CACHED))
    refused = sum(1 for _, r, _ in runs if r.outcome == Outcome.REFUSED)
    escalated_reasons = [
        r.escalated_reason for _, r, _ in runs if r.outcome == Outcome.ESCALATED
    ]
    safety = sum(1 for r in escalated_reasons if is_safety_escalation(r))
    technical = sum(1 for r in escalated_reasons if is_technical_escalation(r))
    avg = sum(m.duration_s for _, _, m in runs) / total
    tokens = sum(m.total_tokens for _, _, m in runs)
    cost = sum(m.cost_rub for _, _, m in runs)
    retries = sum(m.retries for _, _, m in runs)
    print(f"Успешность: {ok}/{total} ({ok * 100 / total:.1f}%) | отказов: {refused} | "
          f"safety-эскалаций: {safety} | тех. ошибок: {technical}")
    print(f"Среднее время: {avg:.2f}с | суммарные токены: {tokens} | "
          f"стоимость: {cost:.4f}₽ | ретраев: {retries}")
    alert = check_error_alert(escalated_reasons, total)
    if alert:
        print(alert)


def check_error_alert(escalated_reasons: list[Optional[str]], total: int,
                      threshold: float = ERROR_ALERT_THRESHOLD) -> Optional[str]:
    """Алерт: доля прогонов с *технической* эскалацией выше порога.

    Safety-эскалации (high_risk) — штатное безопасное поведение агента — не
    считаются ошибкой. Порог по умолчанию 0.2 (20%): при большом числе
    прогонов случайные сбои будут заметнее.
    """
    if total == 0 or threshold <= 0:
        return None
    technical = sum(1 for r in escalated_reasons if is_technical_escalation(r))
    share = technical / total
    if share > threshold:
        return (
            f"[АЛЕРТ] {share * 100:.1f}% прогонов завершилось технической "
            f"эскалацией (порог {threshold * 100:.0f}%) — проверьте LLM и "
            f"конфигурацию; safety-эскалации (high_risk) не учитываются"
        )
    return None


def report(path: str) -> int:
    """Агрегированные метрики по всему логам выполнения (--report)."""
    records = RunLog(path).read()
    if not records:
        print(f"Лог «{path}» пуст или не найден. Сначала выполните прогон: agent.py --demo")
        return 1
    total = len(records)
    by_outcome: dict[str, list[dict]] = {}
    for rec in records:
        by_outcome.setdefault(str(rec.get("outcome", "unknown")), []).append(rec)
    ok = sum(len(v) for k, v in by_outcome.items() if k in ("answered", "answered_cached"))
    print(f"Отчёт по логам выполнения: {path} (всего {total} прогонов)")
    print(f"{'результат':<18}{'прогонов':>9}{'доля':>9}{'среднее время':>16}"
          f"{'токенов':>11}{'стоимость':>13}")
    for outcome in ("answered", "answered_cached", "refused", "escalated"):
        recs = by_outcome.get(outcome)
        if not recs:
            continue
        avg = sum(r.get("duration_s", 0) for r in recs) / len(recs)
        tokens = sum(r.get("tokens", {}).get("total", 0) for r in recs)
        cost = sum(r.get("cost_rub", 0) for r in recs)
        print(f"{outcome:<18}{len(recs):>9}{len(recs) * 100 / total:>8.1f}%"
              f"{avg:>15.2f}с{tokens:>11}{cost:>12.4f}₽")
    for other, recs in by_outcome.items():
        if other in ("answered", "answered_cached", "refused", "escalated"):
            continue
        print(f"{other:<18}{len(recs):>9}{len(recs) * 100 / total:>8.1f}%"
              f"{'—':>16}{'—':>11}{'—':>13}")
    total_dur = sum(r.get("duration_s", 0) for r in records)
    total_tok = sum(r.get("tokens", {}).get("total", 0) for r in records)
    total_ret = sum(r.get("retries", 0) for r in records)
    total_cost = sum(r.get("cost_rub", 0) for r in records)
    budget = sum(1 for r in records if r.get("escalated_reason") == "budget_exceeded")
    escalated_reasons = [
        r.get("escalated_reason") for r in records if r.get("outcome") == "escalated"
    ]
    safety = sum(1 for r in escalated_reasons if is_safety_escalation(r))
    technical = sum(1 for r in escalated_reasons if is_technical_escalation(r))
    print(f"Успешность: {ok}/{total} ({ok * 100 / total:.1f}%) | "
          f"safety-эскалаций: {safety} | тех. ошибок: {technical}")
    print(f"Суммарно: время {total_dur:.2f}с, токенов {total_tok}, стоимость {total_cost:.4f}₽, "
          f"ретраев {total_ret}, эскалаций по бюджету {budget}")
    alert = check_error_alert(escalated_reasons, total)
    if alert:
        print(alert)
    return 0


DEMO_QUERIES: list[tuple[str, str]] = [
    ("Как получить сертификат об окончании курса?",
     "happy path: полный путь + сохранение в память"),
    ("Какая погода в Токио?",
     "нет контекста → ветка refuse"),
    ("Я хочу на вас подать в суд и причинить вред себе.",
     "высокий риск → ветка escalate"),
    ("Как получить сертификат об окончании курса?",
     "повтор вопроса 1 → хит в памяти"),
]


def run_one(workflow: Workflow, query: str, show_trace: bool,
            meter: MeteredChat, run_log: RunLog
            ) -> Optional[tuple[AgentResult, RunMetrics]]:
    """Одиночный прогон: сценарий + вывод + строка метрик + запись в лог."""
    try:
        result, metrics = _execute(query, workflow, meter)
    except openai.APIError as e:
        print(f"[ошибка LLM] {e.__class__.__name__}: {str(e)[:300]}")
        return None
    print_result(result, show_trace)
    print_metrics(metrics)
    log_run(run_log, query, result, metrics)
    return result, metrics


def run_demo(workflow: Workflow, meter: MeteredChat, run_log: RunLog) -> None:
    print(f"Демо: {len(DEMO_QUERIES)} прогона (все ветки графа состояний)")
    runs: list[tuple[str, AgentResult, RunMetrics]] = []
    for i, (query, comment) in enumerate(DEMO_QUERIES, 1):
        print(f"\n===== Прогон {i}/{len(DEMO_QUERIES)}: {comment} =====")
        print(f"Запрос: {query}")
        one = run_one(workflow, query, show_trace=True, meter=meter, run_log=run_log)
        if one is not None:
            runs.append((comment, one[0], one[1]))
    print_metrics_table(runs)


def chat_loop(workflow: Workflow, meter: MeteredChat, run_log: RunLog) -> None:
    print("Интерактивный режим. Выход: 'exit'/'quit'/'выход' или Ctrl-D.")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "выход"):
            break
        run_one(workflow, q, show_trace=True, meter=meter, run_log=run_log)


def _qa_count(memory: GraphMemory) -> int:
    return sum(1 for n in memory.nodes.values() if n.type == "qa")


# --------------------------------------------------------------------------- #
# Самодиагностика
# --------------------------------------------------------------------------- #

HAPPY_QUERY = "Как получить сертификат об окончании курса?"
REFUSE_QUERY = "Какая погода в Токио?"
RISKY_QUERY = "Я хочу на вас подать в суд и причинить вред себе."


def _selftest_env(mode: str, config: Optional[ScenarioConfig] = None,
                  max_calls: int = MAX_LLM_CALLS_PER_RUN) -> dict[str, Any]:
    """Свежая среда для проверки: БЗ, tmp-память, мок Qdrant (TF-IDF), FakeLLM.

    Лог выполнения в selftest не ведётся (кроме отдельной проверки,
    где он создаётся в tmp-каталоге) — реальный logs/runs.jsonl не трогает.
    """
    kb = load_documents(_resolve(KB_FILE))
    qa_dir = tempfile.mkdtemp(prefix="dz6-selftest-")
    qa_path = os.path.join(qa_dir, "qa.json")
    memory = build_runtime_memory(kb, qa_path)
    # Мок Qdrant + детерминированный TF-IDF эмбеддер: поиск осмыслен, сети нет.
    embedder = TfidfEmbedder([f"{d.title} {d.text}" for d in kb])
    qmem = MockQdrantMemory(embedder.embed)
    qmem.upsert_documents(kb)
    raw = FakeLLM(mode)
    # Метрикация: retry фиксированный (2 — не зависит от .env), бюджет — параметр.
    meter = MeteredChat(raw, max_retries=2)
    meter.guard.max_calls = max_calls
    workflow = build_scenario(meter, kb, memory, qmem, qa_path, config or ScenarioConfig())
    return {"chat": raw, "meter": meter, "memory": memory, "qmem": qmem,
            "qa_path": qa_path, "workflow": workflow}


def _load_qa_file(qa_path: str) -> dict[str, Any]:
    # Файл появляется только после первого сохранения — «нет файла» = пустая память.
    if not os.path.exists(qa_path):
        return {"nodes": [], "edges": []}
    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gen_call_count(chat: FakeLLM) -> int:
    """Сколько раз FakeLLM генерировала ответ (системный промпт агента)."""
    return sum(1 for s, _ in chat.calls if "агент поддержки" in s)


def selftest() -> int:
    """14 проверок без LLM, без Qdrant и без сети (FakeLLM + мок Qdrant + TF-IDF)."""
    failures: list[str] = []

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e!r}")
            failures.append(name)

    # 1. Граф валиден: все 13 состояний достижимы, переходов в «фантазии» нет.
    def t1_graph_valid() -> None:
        env = _selftest_env("default")
        results = [
            env["workflow"].run(WorkflowContext(query=HAPPY_QUERY)),
            env["workflow"].run(WorkflowContext(query=HAPPY_QUERY)),  # повтор → хит
        ]
        for mode, query in [("default", REFUSE_QUERY), ("risky", RISKY_QUERY),
                            ("ungrounded", HAPPY_QUERY)]:
            results.append(_selftest_env(mode)["workflow"].run(WorkflowContext(query=query)))
        all_states = set(env["workflow"].states)
        assert len(all_states) == 13, f"ожидается 13 состояний, а их {len(all_states)}"
        visited: set[str] = set()
        for r in results:
            for state_name in r.trace:
                assert state_name in all_states, f"переход в неизвестное состояние {state_name}"
            visited.update(r.trace)
        missing = all_states - visited
        assert not missing, f"недостижимые состояния: {sorted(missing)}"

    # 2. Success path: точный trace (с check_answer), ответ, сохранение в память.
    def t2_success_path() -> None:
        env = _selftest_env("default")
        result, metrics = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        expected_trace = [
            "check_memory", "classify", "retrieve", "check_relevance",
            "generate", "check_answer", "validate", "check_validation",
            "save", "finish",
        ]
        assert result.trace == expected_trace, f"trace: {result.trace}"
        assert result.outcome == Outcome.ANSWERED, result.outcome
        assert result.memory_saved is True
        assert "doc-certificate" in result.sources, f"sources: {result.sources}"
        assert metrics.success is True and metrics.outcome == Outcome.ANSWERED
        expected_cost = round(
            metrics.prompt_tokens * LLM_COST_INPUT_PER_1M_RUB / 1_000_000
            + metrics.completion_tokens * LLM_COST_OUTPUT_PER_1M_RUB / 1_000_000,
            6,
        )
        assert metrics.cost_rub == expected_cost, (
            f"стоимость: {metrics.cost_rub} != {expected_cost}"
        )
        qa_data = _load_qa_file(env["qa_path"])
        qa_nodes = [n for n in qa_data["nodes"] if n["type"] == "qa"]
        assert len(qa_nodes) == 1, f"qa-узлы: {qa_data['nodes']}"
        assert any(
            e["from"] == qa_nodes[0]["id"] and e["to"] == "doc-certificate"
            for e in qa_data["edges"]
        ), f"ребро qa→doc-certificate не найдено: {qa_data['edges']}"

    # 3. Ветка «нет контекста»: отказ, память не пополняется;
    #    отрицательный косинус обрезается до [0, 1].
    def t3_refuse_branch() -> None:
        env = _selftest_env("default")
        result, _ = _execute(REFUSE_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.REFUSED, result.outcome
        assert result.trace == [
            "check_memory", "classify", "retrieve", "check_relevance", "refuse",
        ], f"trace: {result.trace}"
        assert result.memory_saved is False
        assert _load_qa_file(env["qa_path"])["nodes"] == [], "память была пополнена"
        # Отрицательный косинус (теоретически возможен): скор обрезают до
        # [0, 1], ветка — refuse.
        kb = load_documents(_resolve(KB_FILE))
        class _NegScoreMem:
            def search(self, query, top_k):
                return [(kb[0].doc_id, -0.6, {}), (kb[1].doc_id, -0.1, {})]
        wf = build_scenario(env["meter"], kb, env["memory"], _NegScoreMem(),
                            env["qa_path"], ScenarioConfig())
        ctx = WorkflowContext(query=REFUSE_QUERY)
        result = wf.run(ctx)
        assert result.outcome == Outcome.REFUSED, result.outcome
        assert ctx.docs, "документы не сохранены в контекст"
        for d in ctx.docs:
            assert 0.0 <= d.score <= 1.0, f"score {d.score} не в [0, 1]"

    # 4. Провал валидации: ungrounded + retries=1 → grounding_validation_failed,
    #    бюджет не трогается (5 вызовов: classify + 2×(generate+validate) < 6).
    def t4_validation_retry() -> None:
        env = _selftest_env("ungrounded", config=ScenarioConfig(max_validation_retries=1))
        result, metrics = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "grounding_validation_failed", result.escalated_reason
        assert result.trace.count("generate") == 1 + 1, f"trace: {result.trace}"
        assert result.memory_saved is False
        assert metrics.llm_calls == 5, f"бюджет не должен трогаться, вызовов: {metrics.llm_calls}"
        assert _load_qa_file(env["qa_path"])["nodes"] == [], "память была пополнена"

    # 5. Высокий риск: эскалация сразу после классификации, без генерации.
    def t5_high_risk() -> None:
        env = _selftest_env("risky")
        result, _ = _execute(RISKY_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "high_risk", result.escalated_reason
        assert result.trace == ["check_memory", "classify", "escalate"], f"trace: {result.trace}"

    # 6. Хит в памяти: повтор вопроса → ответ из памяти, без новой генерации.
    def t6_memory_hit() -> None:
        env = _selftest_env("default")
        first, _ = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert first.outcome == Outcome.ANSWERED and first.memory_saved
        gen_calls_1 = _gen_call_count(env["chat"])
        second, metrics2 = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert second.outcome == Outcome.ANSWERED_CACHED, second.outcome
        assert second.trace == ["check_memory", "finish_cached"], f"trace: {second.trace}"
        assert _gen_call_count(env["chat"]) == gen_calls_1, "сделана новая генерация при хите"
        assert metrics2.llm_calls == 0, "хит в памяти не должен вызывать LLM"
        assert second.sources == first.sources, f"sources: {second.sources} != {first.sources}"

    # 7. Гард по шагам: при бесконечном ретрае прогон останавливается по MAX_STEPS.
    def t7_step_guard() -> None:
        config = ScenarioConfig(max_validation_retries=10, max_steps=8)
        env = _selftest_env("loop", config)
        result, _ = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "max_steps", result.escalated_reason
        assert len(result.trace) <= 8, f"trace: {result.trace}"

    # 8. Бюджет: ungrounded + MAX_LLM_CALLS_PER_RUN=4 → budget_exceeded,
    #    ровно 4 LLM-вызова, запись с llm_calls=4 в tmp-логе.
    def t8_budget() -> None:
        env = _selftest_env("ungrounded", max_calls=4)
        log_path = os.path.join(tempfile.mkdtemp(prefix="dz6-selftest-log-"), "runs.jsonl")
        run_log = RunLog(log_path)
        result, metrics = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        log_run(run_log, HAPPY_QUERY, result, metrics)
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "budget_exceeded", result.escalated_reason
        assert metrics.llm_calls == 4, f"вызовов: {metrics.llm_calls}"
        assert metrics.budget_violated is True
        records = run_log.read()
        assert records and records[-1]["llm_calls"] == 4, f"лог: {records}"
        assert "cost_rub" in records[-1], f"в записи нет стоимости: {records[-1]}"

    # 9. Retry: flaky (первые 2 вызова — сетевая ошибка) → прогон завершён успешно.
    def t9_retry() -> None:
        env = _selftest_env("flaky")
        result, metrics = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.ANSWERED, result.outcome
        assert metrics.retries == 2, f"ретраев: {metrics.retries}"
        # 3 попытки классификации + generate + validate.
        assert metrics.llm_calls == 5, f"вызовов: {metrics.llm_calls}"

    # 10. check_answer: badformat (doc-fake в источниках) → ретраи →
    #     эскалация answer_check_failed, LLM-валидатор не вызывается;
    #     документ из БЗ, но не из топ-k текущего поиска — тоже отклоняется.
    def t10_check_answer() -> None:
        env = _selftest_env("badformat")
        result, metrics = _execute(HAPPY_QUERY, env["workflow"], env["meter"])
        assert result.outcome == Outcome.ESCALATED, result.outcome
        assert result.escalated_reason == "answer_check_failed", result.escalated_reason
        assert "validate" not in result.trace, f"LLM-валидатор не должен вызываться: {result.trace}"
        # classify + 3×generate (check_answer детерминированно отвергает каждый раз).
        assert metrics.llm_calls == 4, f"вызовов: {metrics.llm_calls}"
        # Retrieval-область: id существует в БЗ, но не получен текущим поиском
        # → ответ отвергается (модель цитирует только то, что видела).
        kb = load_documents(_resolve(KB_FILE))
        all_ids = {d.doc_id for d in kb}
        stale = next(d for d in kb if d.doc_id != "doc-certificate")
        reason = _answer_problem(
            f"Ответ. [источники: {stale.doc_id}]", all_ids, {"doc-certificate"}, 2000)
        assert reason and "не получены текущим поиском" in reason, f"reason: {reason}"
        reason = _answer_problem("Ответ. [источники: doc-fake]", all_ids, all_ids, 2000)
        assert reason and "несуществующие id" in reason, f"reason: {reason}"
        assert _answer_problem(
            "Ответ. [источники: doc-certificate]", all_ids, {"doc-certificate"}, 2000
        ) is None

    # 11. Ротация лога: файл не растёт за max_records, переполненный файл —
    #     целиком в архив .1, новый лог — с новой записи.
    def t11_log_rotation() -> None:
        log_path = os.path.join(tempfile.mkdtemp(prefix="dz6-selftest-rot-"), "runs.jsonl")
        run_log = RunLog(log_path, max_records=3)
        for i in range(4):
            run_log.append({"i": i})
        assert [r["i"] for r in run_log.read()] == [3], "после ротации файл не новый"
        assert [r["i"] for r in RunLog(log_path + ".1").read()] == [0, 1, 2], "архив не полный"
        # Вторая ротация: архив заменяется, а не накапливается.
        for i in range(4, 7):
            run_log.append({"i": i})
        assert [r["i"] for r in run_log.read()] == [6], "вторая ротация не сработала"
        assert [r["i"] for r in RunLog(log_path + ".1").read()] == [3, 4, 5], "архив накопился"
        # Лимит 0 — ротация выключена.
        no_rot = os.path.join(tempfile.mkdtemp(prefix="dz6-selftest-rot-"), "runs.jsonl")
        for i in range(5):
            RunLog(no_rot, max_records=0).append({"i": i})
        assert len(RunLog(no_rot).read()) == 5, "ротация сработала при max_records=0"

    # 12. Алерт: считается доля *технических* эскалаций; safety (high_risk) —
    #     штатное поведение, ошибкой не считается даже при большой доле.
    def t12_error_alert() -> None:
        assert check_error_alert([], 4, 0.2) is None
        assert check_error_alert(["high_risk"], 4, 0.2) is None  # 25%, но safety
        assert check_error_alert(["high_risk", "high_risk"], 4, 0.2) is None  # 50% safety
        assert check_error_alert(["step_error:validate"], 4, 0.2) is not None
        assert check_error_alert(  # 10% technical < 20%
            ["high_risk", "grounding_validation_failed"], 10, 0.2) is None
        assert check_error_alert(  # 30% technical > 20%
            ["high_risk"] + ["max_steps"] * 3, 10, 0.2) is not None
        assert check_error_alert([], 0, 0.2) is None  # пустой лог — алерта нет

    # 13. Классификация эскалаций: high_risk — safety, всё остальное — technical.
    def t13_escalation_kinds() -> None:
        assert is_safety_escalation("high_risk")
        assert not is_safety_escalation("step_error:generate")
        assert not is_safety_escalation(None)
        assert is_technical_escalation("step_error:generate")
        assert is_technical_escalation("budget_exceeded")
        assert is_technical_escalation("max_steps")
        assert is_technical_escalation("classification_failed")
        assert is_technical_escalation("grounding_validation_failed")
        assert is_technical_escalation("answer_check_failed")
        assert not is_technical_escalation("high_risk")
        assert not is_technical_escalation(None)

    # 14. Gold set: файл валиден (каждый факт — подстрока текста ожидаемого
    #     источника в БЗ), harness на FakeLLM считает итоги, факты и источники.
    def t14_gold_set() -> None:
        cases = load_eval_cases(_resolve(EVAL_FILE))
        assert len(cases) >= 4, f"кейсов мало: {len(cases)}"
        docs = {d.doc_id: d for d in load_documents(_resolve(KB_FILE))}
        for c in cases:
            assert c.id and c.query.strip(), f"пустой кейс: {c}"
            if c.expect_source:
                doc = docs.get(c.expect_source)
                assert doc, f"кейс {c.id}: источник {c.expect_source} нет в БЗ"
                for fact in c.expect_facts:
                    assert fact.lower() in doc.text.lower(), (
                        f"кейс {c.id}: факт «{fact}» не найден в тексте {c.expect_source}")
        # Harness на FakeLLM (default): итоги верные, счёт фактов и источников
        # работает (универсальный ответ заглушки факты не содержит — и так и есть).
        env = _selftest_env("default")
        results = run_eval(cases, env["workflow"], env["meter"])
        assert len(results) == len(cases), "результатов не по числу кейсов"
        by_id = {er.case.id: er for er in results}
        assert by_id["certificate"].result.outcome == Outcome.ANSWERED
        assert by_id["off-topic"].result.outcome == Outcome.REFUSED
        cert = by_id["certificate"]
        assert cert.facts_total > 0 and cert.facts_matched < cert.facts_total, (
            f"у FakeLLM факты не должны совпасть: {cert.facts_matched}/{cert.facts_total}")
        assert cert.source_ok, "FakeLLM цитирует retrieved-документы — источник должен найтись"
        assert by_id["off-topic"].passed, "refused-кейс без фактов должен проходить"
        # Юнит match_facts: регистр и пробелы не мешают, отсутствующий факт — в missed.
        m, ms = match_facts("Сертификат выдадут в личном кабинете за 5 рабочих дней",
                            ["личном кабинете", "5 рабочих дней", "возврат"])
        assert m == ["личном кабинете", "5 рабочих дней"] and ms == ["возврат"]

    check("граф валиден: все 13 состояний достижимы, переходов в «фантазии» нет", t1_graph_valid)
    check("success path: точный trace с check_answer, ответ, сохранение в память", t2_success_path)
    check("ветка «нет контекста»: отказ, память не пополняется", t3_refuse_branch)
    check("провал валидации: ретраи исчерпаны → эскалация, бюджет не тронут", t4_validation_retry)
    check("высокий риск: эскалация после классификации, без генерации", t5_high_risk)
    check("хит в памяти: ответ из памяти, без новой генерации и LLM-вызовов", t6_memory_hit)
    check("гард MAX_STEPS: бесконечный ретрай останавливается", t7_step_guard)
    check("бюджет: budget_exceeded, ровно 4 LLM-вызова, запись в лог", t8_budget)
    check("retry: сетевые ошибки пережиты (3 попытки), прогон успешен", t9_retry)
    check("check_answer: галлюцинированный источник → answer_check_failed", t10_check_answer)
    check("ротация лога: runs.jsonl не растёт за лимит, архив <файл>.1", t11_log_rotation)
    check("алерт: учитываются только технические эскалации, safety — нет", t12_error_alert)
    check("классификация эскалаций: high_risk — safety, остальное — technical",
          t13_escalation_kinds)
    check("gold set: факты валидны против БЗ, harness считает качество ответов",
          t14_gold_set)

    if failures:
        print(f"SELF-TEST: {len(failures)} упало: {', '.join(failures)}")
        return 1
    print("SELF-TEST: все проверки пройдены.")
    return 0


# --------------------------------------------------------------------------- #
# Главная ------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="DZ_6: пайплайн тикета поддержки + контрольный слой "
                    "(метрики, бюджет LLM-вызовов, лог выполнения)")
    parser.add_argument("question", nargs="*", help="одиночный вопрос")
    parser.add_argument("--demo", action="store_true",
                        help="4 прогона, покрывающие все ветки + сводная таблица метрик")
    parser.add_argument("--selftest", action="store_true", help="самодиагностика без LLM")
    parser.add_argument("--report", action="store_true",
                        help="агрегированные метрики по логам runs.jsonl")
    parser.add_argument("--eval", action="store_true",
                        help="gold set: оценка качества ответов (нужны LLM и Qdrant)")
    parser.add_argument("--show-trace", action="store_true",
                        help="печатать путь по состояниям графа")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.report:
        sys.exit(report(_resolve(RUNS_LOG_FILE)))

    kb_path = _resolve(KB_FILE)
    qa_path = _resolve(QA_MEMORY_FILE)
    kb = load_documents(kb_path)
    memory = build_runtime_memory(kb, qa_path)
    client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    qmem = make_vector_memory(client, kb)
    raw_chat = make_openai_chat(client)
    meter = MeteredChat(raw_chat)
    workflow = build_scenario(meter, kb, memory, qmem, qa_path, ScenarioConfig())
    run_log = RunLog(_resolve(RUNS_LOG_FILE))
    print(
        f"Агент DZ_6 | БЗ: {len(kb)} документов | память: {_qa_count(memory)} обработанных "
        f"вопросов | модель={LLM_MODEL}"
    )

    try:
        if args.eval:
            sys.exit(run_eval_cli(kb, qmem, raw_chat))
        if args.demo:
            run_demo(workflow, meter, run_log)
        elif args.question:
            run_one(workflow, " ".join(args.question), show_trace=args.show_trace,
                    meter=meter, run_log=run_log)
        else:
            chat_loop(workflow, meter, run_log)
    finally:
        qmem.close()


if __name__ == "__main__":
    main()