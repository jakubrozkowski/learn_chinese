import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

### VARIABLES ###

env = dotenv_values('.env')

if 'OPENAI_API_KEY' in st.secrets:
    env['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']

qdrant_client = QdrantClient(
    url=env["QDRANT_URL"],
    api_key=env["QDRANT_API_KEY"]
)

EMBEDDING_DIM = 1536

EMBEDDING_MODEL = "text-embedding-3-small"

QDRANT_TRANSLATIONS_COLLECTION = "translations"
QDRANT_IMPROVE_COLLECTION = "improve"

AUDIO_TRANSCRIBE_MODEL = 'whisper-1'
OPENAI_TEXT_MODEL = 'gpt-4o-mini'
OPENAI_AUDIO_MODEL = 'gpt-4o-mini-tts'

if "last_saved_translation" not in st.session_state:
    st.session_state["last_saved_translation"] = None
if "last_saved_improve" not in st.session_state:
    st.session_state["last_saved_improve"] = None
if 'translation_input' not in st.session_state:
    st.session_state['translation_input'] = ''
if 'translation_result' not in st.session_state:
    st.session_state['translation_result'] = ''
if 'translation_explanation' not in st.session_state:
    st.session_state['translation_explanation'] = ''
if 'translation_audio' not in st.session_state:
    st.session_state['translation_audio'] = ''
if 'improve_input' not in st.session_state:
    st.session_state['improve_input'] = ''
if 'improve_result' not in st.session_state:
    st.session_state['improve_result'] = ''
if 'improve_explanation' not in st.session_state:
    st.session_state['improve_explanation'] = ''
if 'improve_audio' not in st.session_state:
    st.session_state['improve_audio'] = ''
if 'language' not in st.session_state:
    st.session_state['language'] = 'English'

### FUNCTIONS ###

def get_openai_client():
    return OpenAI(api_key=env['OPENAI_API_KEY'])


def translate(prompt):
    try:
        openai_client = get_openai_client()
        res = openai_client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": 'Jesteś znawcą języka chińskiego...',
                },
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None


def improve(prompt):
    try:
        openai_client = get_openai_client()
        res = openai_client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": 'Jesteś znawcą języka chińskiego. Popraw tekst, aby nie było w nim błędów i brzmiał naturalnie. Wygeneruj tylko poprawiony tekst.',
                },
                {"role": "user", "content": prompt},
            ],
        )
        return res.choices[0].message.content
    except Exception as e:
        st.error(f"Improve error: {str(e)}")
        return None


def explain_improve(prompt, user_input):
    openai_client = get_openai_client()
    res = openai_client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f'Jesteś znawcą języka chińskiego. Użytkownik wpisał zdanie {user_input}. '
                    f'Oceń to zdanie i wytłumacz czy wersja "{prompt}" jest bardziej poprawna i dlaczego. '
                    f'Użyj języka {st.session_state["language"]}.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return res.choices[0].message.content


def explain_translation(prompt):
    openai_client = get_openai_client()
    res = openai_client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f'Jesteś znawcą języka chińskiego. Wytłumacz najtrudniejsze słowa i gramatykę z tekstu poniżej.'
                    f'Użyj języka {st.session_state["language"]}.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return res.choices[0].message.content


def translation_audio(text):
    openai_client = get_openai_client()
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice='onyx',
        response_format="mp3",
        input=text,
    )

    audio_trans_bytes = response.read()
    return audio_trans_bytes

def improve_audio(text):
    openai_client = get_openai_client()
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice='onyx',
        response_format="mp3",
        input=text,
    )

    audio_imp_bytes = response.read()
    return audio_imp_bytes

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_to_database(collection, note_text):
        points_count = qdrant_client.count(
            collection_name=collection,
            exact=True,
        )
        qdrant_client.upsert(
        collection_name= collection,
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector=get_embedding(text=note_text),
                payload={
                    "text": note_text,
                },
            )
        ]
    )
        
def list_notes_from_db(collection_name, query=None):
    if not query:
        notes = qdrant_client.scroll(collection_name=collection_name, limit=3)[0]
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": None,
            })

        return result

    else:
        notes = qdrant_client.search_points(
            collection_name=collection_name,
            query_vector=get_embedding(text=query),
            limit=3,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })

        return result

def translation_changed():
    st.session_state['translation_explanation'] = ''
    st.session_state['translation_audio'] = ''

def improve_changed():
    st.session_state['improve_explanation'] = ''
    st.session_state['improve_audio'] = ''

def language_changed():
    st.session_state['translation_input'] = ''
    st.session_state['translation_result'] = ''
    st.session_state['improve_input'] = ''
    st.session_state['improve_result'] = ''
    st.session_state['improve_explanation'] = ''
    st.session_state['improve_audio'] = ''
    st.session_state['translation_explanation'] = ''
    st.session_state['translation_audio'] = ''

### CALLBACK FUNCTIONS ###

def generate_audio_trans_callback():
    with st.spinner('Generating audio...'):
        st.session_state['translation_audio'] = translation_audio(st.session_state['translation_result'])

def generate_audio_imp_callback():
    with st.spinner('Generating audio...'):
        st.session_state['improve_audio'] = improve_audio(st.session_state['improve_result'])

def generate_translation_explanation_callback():
    with st.spinner('Explaining...'):
        st.session_state['translation_explanation'] = explain_translation(st.session_state['translation_result'])

def generate_improve_explanation_callback():
    with st.spinner('Explaining...'):
        st.session_state['improve_explanation'] = explain_improve(st.session_state['improve_result'], st.session_state['improve_input'])

### MAIN ###

st.session_state['language'] = st.selectbox(
    'Choose your language', ['English', 'Polski', 'Español', '中文'],
    on_change=language_changed
)

if st.session_state['language'] == 'English':
    translator = 'Translator'
    history = 'History'
if st.session_state['language'] == 'Polski':
    translator = 'Tłumacz'
    history = 'Historia'
if st.session_state['language'] == 'Español':
    translator = 'Traductor'
    history = 'Historial'
if st.session_state['language'] == '中文':
    translator = '翻译器'
    history = '历史记录'

add_tab, search_tab = st.tabs([translator, history])

with add_tab:

    if st.session_state.get('translation_result') and \
       st.session_state['translation_result'] != st.session_state['last_saved_translation']:
        add_to_database(QDRANT_TRANSLATIONS_COLLECTION, st.session_state['translation_result'])
        st.session_state['last_saved_translation'] = st.session_state['translation_result']

    if st.session_state.get('improve_result') and \
       st.session_state['improve_result'] != st.session_state['last_saved_improve']:
        add_to_database(QDRANT_IMPROVE_COLLECTION, st.session_state['improve_result'])
        st.session_state['last_saved_improve'] = st.session_state['improve_result']

    # ENGLISH VERSION #

    # Translate

    if st.session_state['language'] == 'English':

        st.title('Chinese Translator')

        st.text_input('Enter your text in Polish:', key='translation_input', on_change=translation_changed)

        if st.session_state['translation_input']:
            st.markdown('#### Translation:')
            with st.spinner('Translating...'):
                st.session_state['translation_result'] = translate(st.session_state['translation_input'])
            st.write(st.session_state['translation_result'])

            st.button('Explain', on_click=generate_translation_explanation_callback, key='b')
            if st.session_state.get('translation_explanation'):
                st.write(st.session_state['translation_explanation'])
        
            st.button('Generate audio', on_click=generate_audio_trans_callback, key='a')
            if st.session_state.get('translation_audio'):
                st.audio(st.session_state['translation_audio'], format='translation_audio/mp3')

        # Improve

        st.markdown('### Do you want to polish your own text?')
        st.session_state['improve_input'] = st.text_input('Enter your text in Chinese:', on_change=improve_changed)

        if st.session_state['improve_input']:
            st.markdown('#### Improved version:')
            with st.spinner('Improving...'):
                st.session_state['improve_result'] = improve(st.session_state['improve_input'])
            st.write(st.session_state['improve_result'])

            st.button('Explain', on_click=generate_improve_explanation_callback, key='c')
            if st.session_state.get('improve_explanation'):
                st.write(st.session_state['improve_explanation'])

            st.button('Generate audio', on_click=generate_audio_imp_callback)
            if st.session_state.get('improve_audio'):
                st.audio(st.session_state['improve_audio'], format='improve_audio/mp3')

    # POLISH VERSION #

    # Translate 

    if st.session_state['language'] == 'Polski':
        st.title('Chiński tłumacz')

        st.text_input('Wpisz tekst po polsku:', key='translation_input', on_change=translation_changed)

        if st.session_state['translation_input']:
            st.markdown('#### Tłumaczenie:')
            with st.spinner('Translating...'):
                st.session_state['translation_result'] = translate(st.session_state['translation_input'])
            st.write(st.session_state['translation_result'])

            st.button('Wytłumacz', on_click=generate_translation_explanation_callback, key='b')
            if st.session_state.get('translation_explanation'):
                st.write(st.session_state['translation_explanation'])
        
            st.button('Wygeneruj audio', on_click=generate_audio_trans_callback, key='a')
            if st.session_state.get('translation_audio'):
                st.audio(st.session_state['translation_audio'], format='translation_audio/mp3')

    # Improve 

        st.markdown('### Chcesz doszlifować własny tekst?')
        st.session_state['improve_input'] = st.text_input('Wpisz tekst po chińsku:', on_change=improve_changed)

        if st.session_state['improve_input']:
            st.markdown('#### Poprawiona wersja:')
            with st.spinner('Improving...'):
                st.session_state['improve_result'] = improve(st.session_state['improve_input'])
            st.write(st.session_state['improve_result'])

            st.button('Wytłumacz', on_click=generate_improve_explanation_callback, key='c')
            if st.session_state.get('improve_explanation'):
                st.write(st.session_state['improve_explanation'])

            st.button('Wygeneruj audio', on_click=generate_audio_imp_callback)
            if st.session_state.get('improve_audio'):
                st.audio(st.session_state['improve_audio'], format='improve_audio/mp3')

    # SPANISH VERSION #

    # Translate

    if st.session_state['language'] == 'Español':
        st.title('Traductor de chino')

        st.text_input('Escribe tu texto en polaco:', key='translation_input', on_change=translation_changed)

        if st.session_state['translation_input']:
            st.markdown('#### Traducción:')
            with st.spinner('Translating...'):
                st.session_state['translation_result'] = translate(st.session_state['translation_input'])
            st.write(st.session_state['translation_result'])

            st.button('Explicar', on_click=generate_translation_explanation_callback, key='b')
            if st.session_state.get('translation_explanation'):
                st.write(st.session_state['translation_explanation'])
        
            st.button('Generar audio', on_click=generate_audio_trans_callback, key='a')
            if st.session_state.get('translation_audio'):
                st.audio(st.session_state['translation_audio'], format='translation_audio/mp3')

        # Improve

        st.markdown('### ¿Quieres pulir tu propio texto?')
        st.session_state['improve_input'] = st.text_input('Escribe tu texto en chino:', on_change=improve_changed)

        if st.session_state['improve_input']:
            st.markdown('#### Versión corregida:')
            with st.spinner('Improving...'):
                st.session_state['improve_result'] = improve(st.session_state['improve_input'])
            st.write(st.session_state['improve_result'])

            st.button('Explicar', on_click=generate_improve_explanation_callback, key='c')
            if st.session_state.get('improve_explanation'):
                st.write(st.session_state['improve_explanation'])

            st.button('Generar audio', on_click=generate_audio_imp_callback)
            if st.session_state.get('improve_audio'):
                st.audio(st.session_state['improve_audio'], format='improve_audio/mp3')

    # CHINESE VERSION #

    if st.session_state['language'] == '中文':
        st.title('提高你的中文水平')

    # Improve 

        st.markdown('### 想提升你自己的句子吗？')
        st.session_state['improve_input'] = st.text_input('请输入你的中文句子：', on_change=improve_changed)

        if st.session_state['improve_input']:
            st.markdown('#### 修改后的版本：')
            with st.spinner('Improving...'):
                st.session_state['improve_result'] = improve(st.session_state['improve_input'])
            st.write(st.session_state['improve_result'])

            st.button('解释', on_click=generate_improve_explanation_callback, key='c')
            if st.session_state.get('improve_explanation'):
                st.write(st.session_state['improve_explanation'])

            st.button('生成音频', on_click=generate_audio_imp_callback)
            if st.session_state.get('improve_audio'):
                st.audio(st.session_state['improve_audio'], format='improve_audio/mp3')

with search_tab:

    # English VERSION #

    if st.session_state['language'] == 'English':
        st.title('History')

        translation_query = st.text_input('Search your translation history')
        for note in list_notes_from_db(collection_name=QDRANT_TRANSLATIONS_COLLECTION, query=translation_query):
            with st.container(border=True):
                st.markdown(note['text'])

        improve_query = st.text_input('Search your improvement history')
        for note in list_notes_from_db(collection_name=QDRANT_IMPROVE_COLLECTION, query=improve_query):
            with st.container(border=True):
                st.markdown(note['text'])


    # POLISH VERSION #

    if st.session_state['language'] == 'Polski':
        st.title('Historia')

        translation_query = st.text_input('Przeszukaj historię swoich tłumaczeń')
        for note in list_notes_from_db(collection_name=QDRANT_TRANSLATIONS_COLLECTION, query=translation_query):
            with st.container(border=True):
                st.markdown(note['text'])
        improve_query = st.text_input('Przeszukaj historię swoich ulepszeń')
        for note in list_notes_from_db(collection_name=QDRANT_IMPROVE_COLLECTION, query=improve_query):
                with st.container(border=True):
                    st.markdown(note['text'])

    # SPANISH VERSION #

    if st.session_state['language'] == 'Español':
        st.title('Historial')

        translation_query = st.text_input('Busca en tu historial de traducciones')
        for note in list_notes_from_db(collection_name=QDRANT_TRANSLATIONS_COLLECTION, query=translation_query):
            with st.container(border=True):
                st.markdown(note['text'])

        improve_query = st.text_input('Busca en tu historial de mejoras')
        for note in list_notes_from_db(collection_name=QDRANT_IMPROVE_COLLECTION, query=improve_query):
            with st.container(border=True):
                st.markdown(note['text'])

    # CHINESE VERSION #

    if st.session_state['language'] == '中文':
        st.title('历史记录')

        translation_query = st.text_input('搜索你的翻译记录')
        for note in list_notes_from_db(collection_name=QDRANT_TRANSLATIONS_COLLECTION, query=translation_query):
            with st.container(border=True):
                st.markdown(note['text'])

        improve_query = st.text_input('搜索你的改进记录')
        for note in list_notes_from_db(collection_name=QDRANT_IMPROVE_COLLECTION, query=improve_query):
            with st.container(border=True):
                st.markdown(note['text'])

