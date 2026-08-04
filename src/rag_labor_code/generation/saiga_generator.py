from pathlib import Path

from llama_cpp import Llama

LLAMA_TYPE = Llama

DEFAULT_SYSTEM_PROMPT = (
    "Ты юридический RAG-ассистент по Трудовому кодексу "
    "Российской Федерации. "
    "Отвечай только на основании предоставленного контекста. "
    "Не выдумывай нормы права. "
    "Сначала определи, какие источники напрямую отвечают "
    "на конкретный вопрос пользователя. "
    "Используй в ответе только эти релевантные источники. "
    "Не включай положения, которые относятся к другой стороне "
    "трудовых отношений, другому основанию прекращения договора "
    "или другой правовой ситуации, даже если они присутствуют "
    "в контексте. "
    "Например, если вопрос касается инициативы работника, "
    "не используй нормы об инициативе работодателя. "
    "Не превращай перекрёстные ссылки внутри статьи в отдельные "
    "основания ответа. "
    "Не перечисляй общие основания, если пользователь спрашивает "
    "об одном конкретном основании или процедуре. "
    "Если информации недостаточно, прямо сообщи об этом. "
    "Указывай использованные источники в формате [Источник N]."
)


def create_saiga_llm(
    model_path: Path,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
    chat_format: str | None = None,
    verbose: bool = False,
) -> Llama:
    """Загружает локальную Saiga в формате GGUF."""

    if not isinstance(model_path, Path):
        raise TypeError("model_path должен быть объектом Path!")

    if not model_path.exists():
        raise FileNotFoundError("Файл модели не найден!")

    if not model_path.is_file():
        raise ValueError("Путь к модели не является файлом!")

    if model_path.suffix.lower() != ".gguf":
        raise ValueError("Файл модели должен иметь расширение .gguf!")

    if n_ctx <= 0:
        raise ValueError("n_ctx должен быть больше 0!")

    if n_gpu_layers < -1:
        raise ValueError("n_gpu_layers должен быть не меньше -1!")

    if n_threads is not None and n_threads <= 0:
        raise ValueError("n_threads должен быть больше 0!")

    if chat_format is not None:
        chat_format = chat_format.strip()

        if not chat_format:
            raise ValueError("chat_format не должен быть пустым!")

    llama = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        chat_format=chat_format,
        verbose=verbose,
    )

    if not isinstance(llama, LLAMA_TYPE):
        raise TypeError("Созданный объект не является Llama!")

    return llama


def build_rag_user_prompt(
    question: str,
    context: str,
) -> str:
    """Создаёт пользовательский RAG-промпт."""

    question = question.strip()

    if not question:
        raise ValueError("Вопрос пустой!")

    context = context.strip()

    if not context:
        raise ValueError("Контекст пустой!")

    return (
        "Вопрос:\n"
        f"{question}\n\n"
        "Контекст:\n"
        f"{context}\n\n"
        "Инструкция:\n"
        "Ответь строго на поставленный вопрос. "
        "Перед формированием ответа мысленно отбери только те "
        "источники, которые непосредственно относятся к предмету "
        "вопроса и указанной в нём стороне трудовых отношений. "
        "Игнорируй тематически близкие, но не отвечающие на вопрос "
        "источники. "
        "Не смешивай инициативу работника, инициативу работодателя, "
        "соглашение сторон и другие самостоятельные основания. "
        "Не добавляй нормы только потому, что они упоминаются "
        "внутри другой статьи. "
        "Каждое существенное юридическое утверждение основывай "
        "на релевантном источнике из контекста."
    )


def generate_answer(
    question: str,
    context: str,
    llm: Llama,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """Генерирует ответ на основании найденного контекста."""

    if not isinstance(llm, LLAMA_TYPE):
        raise TypeError("llm должен быть объектом Llama!")

    system_prompt = system_prompt.strip()

    if not system_prompt:
        raise ValueError("Системный промпт пустой!")

    if max_tokens <= 0:
        raise ValueError("max_tokens должен быть больше 0!")

    if not 0 <= temperature <= 2:
        raise ValueError("temperature должна находиться от 0 до 2!")

    if not 0 < top_p <= 1:
        raise ValueError(
            "top_p должен находиться от 0 исключительно " "до 1 включительно!"
        )

    user_prompt = build_rag_user_prompt(
        question=question,
        context=context,
    )

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )

    if not isinstance(response, dict):
        raise ValueError("Ответ LLM не содержит choices!")

    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise ValueError("Ответ LLM не содержит choices!")

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        raise ValueError("Ответ LLM не содержит message!")

    message = first_choice.get("message")

    if not isinstance(message, dict):
        raise ValueError("Ответ LLM не содержит message!")

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM вернула пустой ответ!")

    return content.strip()
