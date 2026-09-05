"""The catalog of ONNX models this build can serve.

Every entry is pinned to a commit sha and documents its own trade-offs, because
choosing an embedding model is a decision with consequences that are expensive to
reverse: switching models invalidates every vector you have stored.

Quality figures are from two sources. MTEB is the general multilingual benchmark.
SkMTEB (arXiv 2606.13647) evaluates 31 models on Slovak specifically, and its ranking
differs sharply from the global one, so it is quoted where it applies. Both are
starting points: `embedforge model compare` exists so you can settle it on your data.
"""

from embedforge.config import Settings
from embedforge.engine.base import EmbeddingBackend, ModelInfo
from embedforge.engine.onnx_backend import OnnxModelConfig, OnnxTextBackend, Pooling

# E5 models were trained with these exact prefixes. Omitting them measurably hurts
# retrieval, so they are part of the model definition rather than a caller's concern.
E5_QUERY_PREFIX = "query: "
E5_DOCUMENT_PREFIX = "passage: "

# Qwen3-Embedding takes a task description on the query side and nothing on the
# document side. Note there is no space after "Query:" - that is what their own code
# does, and the prompt is part of what the model was trained on.
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# The -instruct variant takes a task description on the query side only.
E5_INSTRUCT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)

CATALOG: list[tuple[ModelInfo, OnnxModelConfig]] = [
    (
        ModelInfo(
            id="e5-small",
            name="intfloat/multilingual-e5-small",
            dimension=384,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "Smallest multilingual model worth serving. 118M parameters, 384-dimensional "
                "vectors, 100+ languages including Slovak."
            ),
            pros=(
                "Fastest option here and the cheapest to store: 384 dimensions is a quarter "
                "of the vector-database cost of a 1024-dimensional model.",
                "Scores 70.32 on SkMTEB (Slovak), beating embeddinggemma-300m despite being "
                "a third of the size.",
                "Comfortably real-time on a modest CPU.",
            ),
            cons=(
                "512-token limit: long documents must be chunked.",
                "Lowest quality of the E5 family; e5-base costs little more and scores higher.",
            ),
            license="MIT",
            size_mb=487,
        ),
        OnnxModelConfig(
            repo_id="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            onnx_file="onnx/model.onnx",
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_QUERY_PREFIX,
            document_prefix=E5_DOCUMENT_PREFIX,
        ),
    ),
    (
        ModelInfo(
            id="e5-small-int8",
            name="intfloat/multilingual-e5-small (int8)",
            dimension=384,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "Dynamically quantized e5-small. Same architecture and vectors, one quarter "
                "of the size and typically around twice as fast on CPU."
            ),
            pros=(
                "135 MB on disk and the lowest memory footprint in the catalog.",
                "Usually the fastest real model on a CPU-only server.",
            ),
            cons=(
                "Quantization costs some accuracy. Compare it against e5-small on your own "
                "data before trusting it: embedforge model compare e5-small e5-small-int8",
                "Not perfectly batch-stable (0.9972): the vector shifts slightly depending "
                "on what shares its batch. The mildest case in this catalog, but real.",
                "Quantization is tuned for AVX-512 VNNI; older CPUs still run it, but gain less.",
            ),
            license="MIT",
            size_mb=135,
        ),
        OnnxModelConfig(
            repo_id="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            onnx_file="onnx/model_qint8_avx512_vnni.onnx",
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_QUERY_PREFIX,
            document_prefix=E5_DOCUMENT_PREFIX,
        ),
    ),
    (
        ModelInfo(
            id="e5-base",
            name="intfloat/multilingual-e5-base",
            dimension=768,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "The balanced default: 278M parameters, 768-dimensional vectors, 100+ "
                "languages. Good Slovak and English quality at a CPU-friendly size."
            ),
            pros=(
                "Scores 72.39 on SkMTEB (Slovak), ahead of gte-multilingual-base and "
                "Qwen3-Embedding-0.6B, both of which are larger.",
                "Sensible starting point: strong enough to keep, cheap enough to run.",
            ),
            cons=(
                "512-token limit: long documents must be chunked.",
                "e5-large-instruct is meaningfully better on Slovak if you can afford it.",
            ),
            license="MIT",
            size_mb=1127,
        ),
        OnnxModelConfig(
            repo_id="intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            onnx_file="onnx/model.onnx",
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_QUERY_PREFIX,
            document_prefix=E5_DOCUMENT_PREFIX,
        ),
    ),
    (
        ModelInfo(
            id="e5-base-int8",
            name="intfloat/multilingual-e5-base (int8)",
            dimension=768,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "Dynamically quantized e5-base. The best quality-per-CPU-cycle option here "
                "if the accuracy cost turns out to be acceptable on your data."
            ),
            pros=(
                "296 MB instead of 1.1 GB, and substantially faster on CPU.",
                "Keeps e5-base's 768 dimensions and its prefix behavior.",
            ),
            cons=(
                "Quantization costs some accuracy; verify with "
                "embedforge model compare e5-base e5-base-int8",
                "Batch stability 0.9861: the same text embeds slightly differently "
                "depending on its batch. Use e5-base where reproducibility matters.",
                "Quantization is tuned for AVX-512 VNNI.",
            ),
            license="MIT",
            size_mb=296,
        ),
        OnnxModelConfig(
            repo_id="intfloat/multilingual-e5-base",
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            onnx_file="onnx/model_qint8_avx512_vnni.onnx",
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_QUERY_PREFIX,
            document_prefix=E5_DOCUMENT_PREFIX,
        ),
    ),
    (
        ModelInfo(
            id="e5-large-instruct",
            name="intfloat/multilingual-e5-large-instruct",
            dimension=1024,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "The best open model for Slovak in the SkMTEB study: 560M parameters, "
                "1024-dimensional vectors, instruction-tuned on the query side."
            ),
            pros=(
                "Top open score on SkMTEB (Slovak) at 77.49, ahead of bge-m3 (74.43) and "
                "close to proprietary gemini-embedding-001 (77.23).",
                "Best choice when retrieval quality matters more than latency.",
            ),
            cons=(
                "2.3 GB download and roughly 2 GB resident; the slowest model here on CPU.",
                "Still limited to 512 tokens despite its size.",
                "1024 dimensions costs the most to store and search.",
            ),
            license="MIT",
            size_mb=2253,
        ),
        OnnxModelConfig(
            repo_id="intfloat/multilingual-e5-large-instruct",
            revision="274baa43b0e13e37fafa6428dbc7938e62e5c439",
            onnx_file="onnx/model.onnx",
            extra_files=("onnx/model.onnx_data",),
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_INSTRUCT_QUERY_PREFIX,
            document_prefix="",
        ),
    ),
    (
        ModelInfo(
            id="bge-m3",
            name="BAAI/bge-m3",
            dimension=1024,
            max_input_tokens=8192,
            symmetric=True,
            description=(
                "Long-context multilingual model: 568M parameters, 8192-token inputs, and "
                "one representation for queries and documents alike."
            ),
            pros=(
                "8192-token context handles whole documents without chunking.",
                "Strong Slovak: 74.43 on SkMTEB, second only to e5-large-instruct here.",
                "Symmetric, so /embed and /query agree and there are no prefixes to get wrong.",
            ),
            cons=(
                "2.3 GB download, and long inputs cost quadratically in attention.",
                "Chunking with a smaller model often beats feeding one long document.",
            ),
            license="MIT",
            size_mb=2285,
        ),
        OnnxModelConfig(
            repo_id="BAAI/bge-m3",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
            onnx_file="onnx/model.onnx",
            extra_files=("onnx/model.onnx_data",),
            pooling=Pooling.CLS,
            max_seq_length=8192,
        ),
    ),
    (
        ModelInfo(
            id="gte-base",
            name="Alibaba-NLP/gte-multilingual-base",
            dimension=768,
            max_input_tokens=8192,
            symmetric=True,
            description=(
                "Long context at a small size: 305M parameters, 8192-token inputs, "
                "768-dimensional vectors, 70+ languages."
            ),
            pros=(
                "The cheapest way to get 8192-token context here: a third of bge-m3's size.",
                "Symmetric, so no prefixes to get wrong.",
            ),
            cons=(
                "71.76 on SkMTEB (Slovak), below e5-base at a larger size.",
                "Fewer languages than the E5 family.",
            ),
            license="Apache-2.0",
            size_mb=1272,
        ),
        OnnxModelConfig(
            repo_id="onnx-community/gte-multilingual-base",
            revision="2edbf5e672aab465f9ed4c154a8b61791c082c69",
            onnx_file="onnx/model.onnx",
            pooling=Pooling.CLS,
            max_seq_length=8192,
        ),
    ),
    (
        ModelInfo(
            id="gte-base-int8",
            name="Alibaba-NLP/gte-multilingual-base (int8)",
            dimension=768,
            max_input_tokens=8192,
            symmetric=True,
            description=(
                "Quantized gte-base. Long context on a small CPU budget: 357 MB for "
                "8192-token inputs."
            ),
            pros=(
                "By far the cheapest long-context option in the catalog.",
                "Symmetric and prefix-free.",
            ),
            cons=(
                "Quantization costs accuracy on top of a model already behind e5-base on Slovak.",
                "Verify with: embedforge model compare gte-base gte-base-int8",
            ),
            license="Apache-2.0",
            size_mb=357,
        ),
        OnnxModelConfig(
            repo_id="onnx-community/gte-multilingual-base",
            revision="2edbf5e672aab465f9ed4c154a8b61791c082c69",
            onnx_file="onnx/model_int8.onnx",
            pooling=Pooling.CLS,
            max_seq_length=8192,
        ),
    ),
    (
        ModelInfo(
            id="jina-v3",
            name="jinaai/jina-embeddings-v3",
            dimension=1024,
            max_input_tokens=8192,
            symmetric=False,
            description=(
                "572M parameters, 8192-token context, and the second-best Slovak score "
                "of any open model. Selects its task with a LoRA adapter rather than a "
                "text prefix."
            ),
            pros=(
                "75.10 on SkMTEB (Slovak), ahead of bge-m3 and behind only "
                "e5-large-instruct among open models.",
                "Best on semantic similarity in that study (89.82), from an explicit "
                "similarity-training objective.",
                "Long context and strong retrieval in one model.",
            ),
            cons=(
                "CC-BY-NC-4.0: non-commercial only. Fine for private use, a licence trap "
                "if this ever becomes a product.",
                "2.3 GB download and slow on CPU.",
                "Task selection happens inside the graph, so one batch means one task - "
                "which the engine already guarantees.",
            ),
            license="CC-BY-NC-4.0",
            size_mb=2310,
        ),
        OnnxModelConfig(
            repo_id="jinaai/jina-embeddings-v3",
            revision="ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
            onnx_file="onnx/model.onnx",
            extra_files=("onnx/model.onnx_data",),
            pooling=Pooling.MEAN,
            max_seq_length=8192,
            # Indices into the model's `lora_adaptations`: retrieval.query and
            # retrieval.passage. The adapter replaces the text prefix entirely, so
            # prepending an instruction as well would corrupt the input.
            query_task_id=0,
            document_task_id=1,
        ),
    ),
    (
        ModelInfo(
            id="e5-sk-large",
            name="slovak-nlp/e5-sk-large",
            dimension=1024,
            max_input_tokens=512,
            symmetric=False,
            description=(
                "multilingual-e5-large with its vocabulary trimmed from 250k to 60k "
                "tokens for Slovak: 35% smaller at 365M parameters, and better on Slovak "
                "than models half a gigabyte larger."
            ),
            pros=(
                "74.70 on SkMTEB, beating bge-m3 (568M) at two-thirds the size and "
                "roughly matching OpenAI's text-embedding-3-large.",
                "The best size-to-quality trade here for Slovak-dominant traffic.",
                "MIT licensed, from the authors of the Slovak benchmark itself.",
            ),
            cons=(
                "No published ONNX build, so the first download exports one locally: "
                "several minutes, uv required, and about 3 GB of disk while it runs.",
                "The trimmed vocabulary is the point and the cost - other languages "
                "degrade. Use the plain E5 models for mixed-language corpora.",
                "512-token limit.",
            ),
            license="MIT",
            size_mb=1462,
        ),
        OnnxModelConfig(
            repo_id="slovak-nlp/e5-sk-large",
            revision="1e67c6dd72ed42620055e168abe88ca49ba7b7da",
            export_from="slovak-nlp/e5-sk-large",
            # optimum writes the export flat, unlike the nested layout of a published build.
            onnx_file="model.onnx",
            pooling=Pooling.MEAN,
            max_seq_length=512,
            query_prefix=E5_QUERY_PREFIX,
            document_prefix=E5_DOCUMENT_PREFIX,
        ),
    ),
    (
        ModelInfo(
            id="qwen3-0.6b",
            name="Qwen/Qwen3-Embedding-0.6B",
            dimension=1024,
            max_input_tokens=32768,
            symmetric=False,
            description=(
                "Decoder-style embedding model with a 32k context, instruction-tuned on "
                "the query side. The 8B sibling topped the MTEB multilingual leaderboard."
            ),
            pros=(
                "32k context, by far the longest here - whole documents, not chunks.",
                "Perfectly batch-stable, unlike every quantized build in this catalog.",
                "Apache-2.0, and a well-supported family.",
            ),
            cons=(
                "70.53 on SkMTEB (Slovak) - below e5-base, which is less than half its "
                "size. The global ranking does not transfer to Slovak.",
                "2.4 GB and 28 decoder layers per call: really a GPU model.",
                "No usable quantized build: the published int8 export changes retrieval "
                "rankings depending on batch composition, so it is not in this catalog.",
            ),
            license="Apache-2.0",
            size_mb=2411,
        ),
        OnnxModelConfig(
            repo_id="onnx-community/Qwen3-Embedding-0.6B-ONNX",
            revision="c25a394dd583836952667c12f008335071b3f43d",
            onnx_file="onnx/model.onnx",
            extra_files=("onnx/model.onnx_data",),
            pooling=Pooling.LAST_TOKEN,
            max_seq_length=32768,
            query_prefix=QWEN3_QUERY_PREFIX,
            document_prefix="",
        ),
    ),
]

ONNX_CONFIGS: dict[str, OnnxModelConfig] = {info.id: config for info, config in CATALOG}


def build_backend(settings: Settings, info: ModelInfo, config: OnnxModelConfig) -> EmbeddingBackend:
    return OnnxTextBackend(info, config, settings)
