import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

_SILENT = logging.CRITICAL + 1

logging.root.setLevel(_SILENT)
for _h in logging.root.handlers[:]:
    logging.root.removeHandler(_h)
logging.root.addHandler(logging.NullHandler())

for _noisy in ("pypdf", "pypdf._cmap", "pypdf._page",
               "huggingface_hub", "huggingface_hub.utils._http",
               "transformers", "sentence_transformers",
               "langchain", "langchain_core", "chromadb"):
    logging.getLogger(_noisy).setLevel(_SILENT)

import glob
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


persist_directory = "./data_db"

# ------------------------------------------------------------
# CARGAR EL MODELO DE EMBEDDINGS
# ------------------------------------------------------------
# Se carga siempre, tanto para crear la DBV como para consultar.

print("\n Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

print("[INFO] Embedding model loaded successfully")
print("[INFO] Model: all-mpnet-base-v2")


# ------------------------------------------------------------
# 1, 2 y 4. CARGAR DATOS Y CREAR BASE VECTORIAL (solo si no existe)
# ------------------------------------------------------------
# Si data_db ya existe se reutiliza directamente,
# evitando re-indexar todos los documentos en cada ejecución.

if os.path.exists(persist_directory):
    # ── Camino rápido: cargar la DB ya persistida ───────────
    print(f"\n{'─'*60}")
    print(f"[INFO] Base de datos encontrada en '{persist_directory}'")
    print(f"[INFO] Cargando DB existente... (se omite re-indexación)")
    print(f"{'─'*60}")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    print("[OK]   Vector database cargada correctamente")
    print(f"[OK]   Ubicación: {persist_directory}")

else:
    # ── Carga, división e indexación ───────
    print(f"\n{'─'*60}")
    print("[AVISO] No se encontró data_db. Construyendo desde cero...")
    print(f"{'─'*60}")

    # PASO 1 — Cargar archivos
    print("\n[PASO 1] Cargando archivos desde la carpeta 'data'...")

    data_dir = "data"
    raw_documents = []

    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if not os.path.isfile(file_path):
            continue
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"  [WARNING] Formato no soportado: {file_path}")
                continue

            # Suprimir salida de librerías ruidosas durante la carga
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                docs = loader.load()
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr

            raw_documents.extend(docs)
            print(f"  [OK] {len(docs):>4} páginas/docs  ←  {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {file_path}: {e}")

    print(f"[INFO] Total de documentos crudos cargados: {len(raw_documents)}")

    # PASO 2 — Dividir en fragmentos (chunks)
    print("\n[PASO 2] Dividiendo documentos en fragmentos (chunks)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50
    )

    documents = text_splitter.split_documents(raw_documents)

    print(f"[INFO] Fragmentos (chunks) generados: {len(documents)}")
    print(f"[INFO] Tamaño de chunk: 150 chars | Overlap: 50 chars")

    # PASO 4 — Crear y persistir la base vectorial
    print("\n[PASO 4] Creando la base vectorial en disco...")
    print(f"[INFO] Destino: {persist_directory}")

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print("[OK]   Base vectorial creada y guardada en disco")
    print(f"[OK]   Ubicación: {persist_directory}")
    print(f"[OK]   Total de chunks indexados: {len(documents)}")
    print(f"{'─'*60}")



# ------------------------------------------------------------
# 5. CONSULTA DEL USUARIO
# ------------------------------------------------------------
# Esta consulta será convertida en embedding y se usará para
# buscar documentos similares en la base vectorial.

query = "¿en que fecha realice el ultimo analisis?" # Consulta definida directamente

print("\n[PASO 5] Processing user query...")
print(f"[QUERY] {query}")


# ------------------------------------------------------------
# 6. BÚSQUEDA VECTORIAL
# ------------------------------------------------------------
# Se calcula el embedding de la consulta y se buscan
# los documentos más cercanos en el espacio vectorial.

print("\n[PASO 6] Performing vector similarity search...")

results = vectorstore.similarity_search_with_score(query, k=20)

print("[INFO] Similarity search completed")
print(f"[INFO] Retrieved documents: {len(results)}")


# ------------------------------------------------------------
# 7. MOSTRAR RESULTADOS — TOP 5 MÁS RELEVANTES
# ------------------------------------------------------------
# Se ordenan los resultados por score (distancia L2 ascendente),
# se deduplicán y se muestran los 5 mejores.

TOP_K = 5

# Ordenar por score ascendente: menor distancia = mayor similitud
results_sorted = sorted(results, key=lambda x: x[1])

seen_contents = set()
unique_results = []

for doc, score in results_sorted:
    content_stripped = doc.page_content.strip()
    if content_stripped not in seen_contents:
        seen_contents.add(content_stripped)
        unique_results.append((doc, score))
    if len(unique_results) == TOP_K:
        break

print(f"\n{'='*70}")
print(f"  TOP {TOP_K} RESULTADOS ")
print(f"  Query: \"{query}\"")
print(f"  Resultados únicos encontrados: {len(unique_results)}")
print(f"{'='*70}\n")

for i, (doc, score) in enumerate(unique_results, 1):
    source   = doc.metadata.get("source", "Desconocido")
    page     = doc.metadata.get("page", None)
    page_info = f" | Pág. {page + 1}" if page is not None else ""

    # Rating visual: 5 estrellas para score ~0, menos estrellas conforme sube
    stars = max(1, 5 - round(score * 2))
    stars_str = "★" * stars + "☆" * (5 - stars)

    print(f"  #{i}  [{stars_str}]  Distancia: {score:.4f}  {'← mejor resultado' if i == 1 else ''}")
    print(f"  Fuente: {os.path.basename(source)}{page_info}")
    print(f"  {'-'*66}")
    text = doc.page_content.strip()
    if len(text) > 700:
        text = text[:700] + " ... [truncado]"
    print(text)
    print()

print(f"{'='*70}")


# ------------------------------------------------------------
# 8. GENERACIÓN DE RESPUESTA CON LLM 
# ------------------------------------------------------------
# Se utiliza el modelo Qwen2.5-0.5B para generar una respuesta
# de forma rápida y con muy poco consumo de recursos.

print("\n[PASO 8] Loading Qwen2.5-0.5B-Instruct...")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto", 
)

# Configurar el pipeline de generación (solo devuelve la respuesta)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    return_full_text=False
)

# Preparar el contexto combinando los fragmentos encontrados
context_text = "\n\n".join([doc.page_content for doc, score in unique_results])

# Estructurar la conversación para el modelo
messages = [
    {
        "role": "system",
        "content": "Utiliza el contexto proporcionado para responder detalladamente. Explica los conceptos Solo basándote en la información proporcionada." 
        # Aqui es donde se le dice al modelo como debe comportarse
    },
    {
        "role": "user",
        "content": f"CONTEXTO:\n{context_text}\n\nPREGUNTA:\n{query}"
        # Aqui es donde se le da la informacion que el modelo debe usar, junto a la pregunta respectivamente
    }
]

# Aplicar el chat template oficial del modelo (esto oculta los tokens técnicos)
final_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generar la respuesta (sin incluir el prompt en el resultado)
output = pipe(final_prompt)
response = output[0]['generated_text'].strip()


response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

print(f"\n{'='*70}")
print("  RESULTADO DEL LLM ")
print(f"{'='*70}")
print(f"  PREGUNTA: {query}")
print(f"  {'-'*66}")
print(f"  RESPUESTA:")
print(f"  {response}")
print(f"{'='*70}")


