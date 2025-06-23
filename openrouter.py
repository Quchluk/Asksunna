import streamlit as st
import requests
import os
from datetime import datetime
import json
import re

# Настройка страницы
st.set_page_config(
    page_title="Claude Coder Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS для улучшения внешнего вида
st.markdown("""
<style>
    .stTextArea > div > div > textarea {
        font-family: 'Courier New', monospace;
    }
    .code-block {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #4CAF50;
        margin: 10px 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 3px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)


class ClaudeAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Claude Coder Assistant",
            "Content-Type": "application/json"
        }

    def get_models(self):
        """Получить список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers)
            if response.status_code == 200:
                models = response.json()['data']
                claude_models = [model for model in models if 'claude' in model['id'].lower()]
                return claude_models
            return []
        except Exception as e:
            st.error(f"Ошибка получения моделей: {e}")
            return []

    def send_message(self, messages, model="anthropic/claude-3.5-sonnet", temperature=0.1, max_tokens=4000):
        """Отправить сообщение Claude"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания ответа")
            return None
        except Exception as e:
            st.error(f"Ошибка при отправке запроса: {e}")
            return None


def extract_code_blocks(text):
    """Извлечь блоки кода из текста"""
    # Паттерн для поиска блоков кода с языком
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def format_message(content, is_user=True):
    """Форматировать сообщение для отображения"""
    message_class = "user-message" if is_user else "assistant-message"
    role = "👤 Вы" if is_user else "🤖 Claude"

    # Проверяем наличие блоков кода
    code_blocks = extract_code_blocks(content)

    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{role}:</strong>
    </div>
    """, unsafe_allow_html=True)

    if code_blocks:
        # Разбиваем текст на части до, между и после блоков кода
        parts = re.split(r'```\w*\n.*?```', content, flags=re.DOTALL)
        code_pattern = r'```(\w+)?\n(.*?)```'
        codes = re.finditer(code_pattern, content, re.DOTALL)

        part_idx = 0
        for match in re.finditer(code_pattern, content, re.DOTALL):
            # Показываем текст до блока кода
            if part_idx < len(parts) and parts[part_idx].strip():
                st.markdown(parts[part_idx])

            # Показываем блок кода
            language = match.group(1) if match.group(1) else 'text'
            code = match.group(2)

            col1, col2 = st.columns([10, 1])
            with col1:
                st.code(code, language=language)
            with col2:
                if st.button("📋", key=f"copy_{hash(code)}", help="Копировать код"):
                    st.code(code, language='text')
                    st.success("Код скопирован!")

            part_idx += 1

        # Показываем оставшийся текст
        if part_idx < len(parts) and parts[part_idx].strip():
            st.markdown(parts[part_idx])
    else:
        st.markdown(content)


def main():
    st.title("🤖 Claude Coder Assistant")
    st.markdown("*Ваш помощник по программированию на базе Claude Sonnet*")

    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")

        # API ключ
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            api_key = st.text_input("OpenRouter API Key:", type="password",
                                    help="Получите ключ на https://openrouter.ai/")

        if api_key:
            claude_api = ClaudeAPI(api_key)

            # Выбор модели
            models = claude_api.get_models()
            if models:
                model_options = [model['id'] for model in models if 'sonnet' in model['id'].lower()]
                if not model_options:
                    model_options = [model['id'] for model in models]

                selected_model = st.selectbox(
                    "Модель:",
                    model_options,
                    index=0 if model_options else None
                )
            else:
                selected_model = "anthropic/claude-3.5-sonnet"
                st.warning("Не удалось загрузить список моделей. Используется модель по умолчанию.")

            # Параметры
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.slider("Max tokens:", 1000, 8000, 4000, 500)

            # Шаблоны для кодинга
            st.header("📝 Шаблоны запросов")
            templates = {
                "Объяснить код": "Объясни, как работает этот код:\n\n```\n{code}\n```",
                "Найти ошибки": "Найди и исправь ошибки в этом коде:\n\n```\n{code}\n```",
                "Оптимизировать": "Оптимизируй этот код:\n\n```\n{code}\n```",
                "Добавить комментарии": "Добавь подробные комментарии к этому коду:\n\n```\n{code}\n```",
                "Конвертировать язык": "Конвертируй этот код с {from_lang} на {to_lang}:\n\n```\n{code}\n```"
            }

            selected_template = st.selectbox("Выберите шаблон:", list(templates.keys()))

            if st.button("📋 Использовать шаблон"):
                st.session_state.use_template = templates[selected_template]

            # Очистка истории
            if st.button("🗑️ Очистить историю"):
                st.session_state.messages = []
                st.rerun()

        else:
            st.error("Необходимо указать API ключ OpenRouter!")
            st.stop()

    # Инициализация истории сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Отображение истории чата
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            format_message(message["content"], message["role"] == "user")

    # Форма для ввода сообщения
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])

        with col1:
            user_input = st.text_area(
                "Ваш вопрос:",
                value=st.session_state.get('use_template', ''),
                height=100,
                placeholder="Например: Напиши функцию для сортировки массива на Python"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("📤 Отправить", use_container_width=True)

        # Дополнительные опции
        col3, col4 = st.columns(2)
        with col3:
            include_context = st.checkbox("Включить контекст предыдущих сообщений", value=True)
        with col4:
            code_focus = st.checkbox("Фокус на коде", value=True)

    # Очистка шаблона после использования
    if 'use_template' in st.session_state:
        del st.session_state.use_template

    # Обработка отправки сообщения
    if submit_button and user_input.strip():
        # Добавляем системный промпт для фокуса на коде
        system_message = {
            "role": "system",
            "content": """Ты - эксперт-программист и наставник. Твоя задача:
1. Писать чистый, понятный и эффективный код
2. Объяснять сложные концепции простым языком
3. Предлагать лучшие практики и оптимизации
4. Всегда включать примеры кода в свои ответы
5. Форматировать код с указанием языка программирования
6. Быть точным и конкретным в объяснениях

Отвечай на русском языке, но код и комментарии к коду пиши на английском."""
        }

        # Подготавливаем сообщения для API
        api_messages = []
        if code_focus:
            api_messages.append(system_message)

        if include_context:
            # Включаем последние 10 сообщений для контекста
            context_messages = st.session_state.messages[-10:]
            for msg in context_messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Добавляем текущее сообщение пользователя
        api_messages.append({
            "role": "user",
            "content": user_input
        })

        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })

        # Показываем сообщение пользователя
        format_message(user_input, True)

        # Отправляем запрос к API
        with st.spinner("🤔 Claude думает..."):
            response = claude_api.send_message(
                api_messages,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens
            )

        if response and 'choices' in response:
            assistant_message = response['choices'][0]['message']['content']

            # Добавляем ответ в историю
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message,
                "timestamp": datetime.now()
            })

            # Показываем ответ
            format_message(assistant_message, False)

            # Информация об использовании токенов
            if 'usage' in response:
                usage = response['usage']
                st.sidebar.info(f"""
                📊 **Использование токенов:**
                - Вход: {usage.get('prompt_tokens', 0)}
                - Выход: {usage.get('completion_tokens', 0)}
                - Всего: {usage.get('total_tokens', 0)}
                """)

            st.rerun()


if __name__ == "__main__":
    main()
