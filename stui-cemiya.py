# Descripción: Aplicación para detectar odio, violencia, lenguaje ofensivo o contenido inapropiado en un vídeo.
# La aplicación divide el vídeo en escenas y transcribe el audio de cada escena.

import streamlit as st
#from utilities.icon import page_icon
import scenedetect
import os
from ast import literal_eval
import requests
import json
import datetime
from openai import OpenAI
import base64
import pandas as pd
import io
from urllib.parse import urlparse
import subprocess

# global variables
url_qwen = 'http://155.210.153.36:8000/v1'
model = 'Qwen/Qwen2.5-VL-7B-Instruct'
url_whisper = 'http://gtc2pc9.cps.unizar.es:8000/transcribediarize'
archivo_audio = 'grabacion.wav' 
parametros = {
   'language': 'es',
   'model_size': 'large-v3',
   'word_timestamps': 'true'
}
supportedLanguages = ['es', 'en', 'ca', 'zh', 'de', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'nl', 'ar', 'sv', 'it',
                      'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr',
                      'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk',
                      'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo',
                      'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn',
                      'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jv', 'su']

terminos_base="""Violencia física, LGBTQ+fobia (homofobia, transfobia, bifobia),\
Machismo (sexismo, misoginia), Racismo y xenofobia, Antigitanismo, Islamofobia, Antisemitismo, Aspectismo.
Cualquier sospecha de que el vídeo o el audio tiene unos de estos términos indícalo en la descripción y clasifícalo de acuerdo al término detectado.
"""

# Function to convert a dictionary to a XLSX data format
def dict_to_excel(data: dict, key_column_name: str = 'Key', sheet_name: str = 'Sheet1', engine: str = 'openpyxl') -> bytes:
    """
    Transforms a dictionary structure into an Excel file and returns the Excel data as bytes.

    Args:
        data (dict): The dictionary to transform.
        key_column_name (str, optional): The column name for keys of input dict. Defaults to 'Key'.
        sheet_name (str, optional):  Name of the Excel sheet. Defaults to 'Sheet1'.
        engine (str, optional):  The Excel writer engine to use. Defaults to 'openpyxl'. 'xlsxwriter' is another option.
                                Make sure the engine is installed (`pip install openpyxl` or `pip install xlsxwriter`).

    Returns:
        bytes: The Excel data as bytes.

    Raises:
        TypeError: If the input 'data' is not a dictionary.
        ValueError: If the 'data' dictionary is empty.
        ImportError: If the specified engine ('openpyxl' or 'xlsxwriter') is not installed.
    """

    if not isinstance(data, dict):
        raise TypeError("Input 'data' must be a dictionary.")
    if not isinstance(sheet_name, str):
        raise TypeError("Input 'sheet_name' must be a string")
    if not data:
        raise ValueError("Input 'data' dictionary cannot be empty.")

    try:
        # Case 1: Dictionary where values are dictionaries (convert to DataFrame easily)
        if all(isinstance(value, dict) for value in data.values()):
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = key_column_name

        # Case 2: Dictionary where values are lists of dictionaries (more complex)
        elif all(isinstance(value, list) and all(isinstance(item, dict) for item in value) for value in data.values()):
            rows = []
            for key, value_list in data.items():
                for item in value_list:
                    row = {key_column_name: key, **item}
                    rows.append(row)
            df = pd.DataFrame(rows)

        # Case 3: Dictionary where values are simple values (convert to series)
        elif all(not isinstance(value, (dict, list)) for value in data.values()):
            series = pd.Series(data)
            df = series.to_frame(name='Value')
            df.index.name = key_column_name


        else:
            raise ValueError("Unsupported dictionary structure. Values must be all dictionaries, lists of dictionaries, or simple values.")


        excel_buffer = io.BytesIO()  # Create an in-memory bytes buffer

        with pd.ExcelWriter(excel_buffer, engine=engine) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False if (all(isinstance(value, list) and all(isinstance(item, dict) for item in value) for value in data.values())) else True )  #index False only for case 2

        excel_data = excel_buffer.getvalue()  # Get the Excel data as bytes
        return excel_data

    except ImportError as e:
        print(f"Error: The '{engine}' engine is not installed. Please install it (e.g., 'pip install openpyxl').")
        raise

    except Exception as e:
        print(f"Error converting dictionary to Excel data: {e}")
        raise

# Function to convert a dictionary to a CSV file

def dict_to_csv(data: dict, key_column_name: str = 'Key'):
    """
    Transforms a dictionary structure into a Pandas DataFrame and then saves it as a CSV file.

    Args:
        data (dict): The dictionary to transform. The keys of the dictionary will become the index or a column
                     depending on the structure of the values, and the values will become the columns or data rows.
                     The values can be dictionaries or lists of dictionaries.
        key_column_name (str, optional): If the dictionary values are dictionaries, this parameter determines
                                       the column name to use for the keys of the input dictionary. Defaults to 'Key'.

    Raises:
        TypeError: If the input 'data' is not a dictionary or 'csv_filepath' is not a string.
        ValueError: If the 'data' dictionary is empty.
    """

    if not isinstance(data, dict):
        raise TypeError("Input 'data' must be a dictionary.")
    if not data:
        raise ValueError("Input 'data' dictionary cannot be empty.")

    try:
        # Case 1: Dictionary where values are dictionaries (convert to DataFrame easily)
        if all(isinstance(value, dict) for value in data.values()):
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = key_column_name
            csv_output = df.to_csv(index=True,sep=";",encoding='utf-8')

        # Case 2: Dictionary where values are lists of dictionaries (more complex)
        elif all(isinstance(value, list) and all(isinstance(item, dict) for item in value) for value in data.values()):
            rows = []
            for key, value_list in data.items():
                for item in value_list:
                    row = {key_column_name: key, **item}  # Combine key and item dictionary
                    rows.append(row)
            df = pd.DataFrame(rows)
            csv_output = df.to_csv(index=False,sep=";",encoding='utf-8')  # No index in this case

        # Case 3: Dictionary where values are simple values (convert to series)
        elif all(not isinstance(value, (dict, list)) for value in data.values()):
            series = pd.Series(data)
            df = series.to_frame(name='Value')  # Convert series to a DataFrame
            df.index.name = key_column_name
            csv_output = df.to_csv(index=True,sep=";",encoding='utf-8')

        else:
            raise ValueError("Unsupported dictionary structure. Values must be all dictionaries, lists of dictionaries, or simple values.")

        print(f"Successfully converted dictionary to CSV")
        return csv_output

    except Exception as e:
        print(f"Error converting dictionary to CSV: {e}")
        raise


# encode in base64 a mp4 file from a local file

def encode_base64_mp4_from_localfile(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""
    with open(content_url, "rb") as f:
        result = "data:video/base64,"+base64.b64encode(f.read()).decode('utf-8')
    return result

# split the video into scenes

def split_video(video_path,scene_path):
    # Create a VideoManager object and perform scene detection
    video_manager = scenedetect.VideoManager([video_path])
    scene_manager = scenedetect.SceneManager()
    scene_manager.add_detector(scenedetect.detectors.AdaptiveDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()

    # Perform scene detection on the video file
    total_frames=scene_manager.detect_scenes(frame_source=video_manager)

    # Obtain the list of detected scenes
    scene_list = scene_manager.get_scene_list(base_timecode,start_in_scene=True)
    

    print(scene_list)
    print(scene_path)
    if len(scene_list) == 0:
        
        print("No scenes detected.")
        print("Duration:", total_frames/video_manager.get_framerate())
        # video duration in timecode datetime.timedelta(seconds=segment['start'])
        video_duration = datetime.timedelta(seconds=total_frames/video_manager.get_framerate())
        print(video_duration)
        # todo el video es una escena
        scene_list = None
    else:
            # Remove scenes with less than 1 second
        scene_list = [scene for scene in scene_list if scene[1].get_frames() - scene[0].get_frames() > video_manager.get_framerate()]
        print("List of scenes obtained:")
        for i, scene in enumerate(scene_list):
            print(
                "    Scene %2d: Start %s / Frame %d, End %s / Frame %d"
                % (
                    i + 1,
                    scene[0].get_timecode(),
                    scene[0].get_frames(),
                    scene[1].get_timecode(),
                    scene[1].get_frames(),
                )
            )

    # save the scene videos to a file

    if scene_list:
        scenedetect.video_splitter.split_video_ffmpeg(video_path, scene_list, output_dir="escenas")
    else:
        # save the video to the scenes folder
        scene_video = f"escenas/{scene_path}-Scene-001.mp4"
        os.rename(video_path, scene_video)
        scene_list = [(base_timecode, video_duration)]
    video_manager.release()
    return scene_list



st.set_page_config(
    page_title="Cemiya Project",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)
def clear_text():
    st.session_state.image_url = ""
    st.session_state.video = False

@st.cache_resource
def qwen_client():
    return OpenAI(
        base_url=url_qwen,
        api_key="EMPTY",
    )   

def get_description(model, client, messages):

#print("Chat completion input:", message)
## Use video url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=messages,
        model=model,
        max_completion_tokens=1024,
    )

    result = chat_completion_from_url.choices[0].message.content
       
    try:
        output=literal_eval(result)
    except:
        output=result
    return output


def transcribe_video(video_path):
    global archivo_audio
    archivo_audio = f"temp_audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    transcripcion=[]
    print(video_path)
# extract the audio from the video
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-y",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        archivo_audio
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True) # check=True lanza excepción si error
        print("Salida de ffmpeg:", result.stdout)
        print("Error de ffmpeg:", result.stderr)  # Imprime los errores también
        print("Conversión exitosa.")
        # systemcall=f"ffmpeg -i {video_path} -y -vn -acodec pcm_s16le -ar 16000 -ac 2 {archivo_audio}"
        # os.system(systemcall)
        files = {'audio_data': open(archivo_audio, 'rb')}
        # Realizar la llamada POST con requests
        respuesta = requests.post(url_whisper, files=files, data=parametros)
        print(respuesta.text)
        # Remove the temporary audio file
        if os.path.exists(archivo_audio):
            os.remove(archivo_audio)
        out=json.loads(respuesta.text)    
        for segment in out['segments']:
            # cambia el formato de tiempo de segundos a HH:MM:SS.mmm
            x=datetime.timedelta(seconds=segment['start'])
            # comprobar si lleva milisegundos
            if '.' in str(x):
                segment['start']=str(x)  
            else:
                segment['start']=str(x)+'.000'
            x=datetime.timedelta(seconds=segment['end'])
            if '.' in str(x):
                segment['end']=str(x)  
            else:           
                segment['end']=str(x)+'.000'
        
            # si existe el campo speaker
            if 'speaker' in segment:
                transcripcion.append([segment['start'], segment['end'], '<S#'+segment['speaker'].replace("SPEAKER_","")+">",segment['text']])
            else:
                transcripcion.append([segment['start'], segment['end'],'<S#>',segment['text']])
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar ffmpeg: {e}")
        print("Salida de ffmpeg (error):", e.stderr)
        # Manejar el error (mostrar mensaje, registrar, etc.)
        transcripcion=""

    if len(transcripcion)==0:
        transcripcion=[['00:00:00.000', '00:00:00.000', '<S#0>', "No se ha detectado ninguna voz en la grabación"]]
    # cambiar el formato de la transcripción a JSON
    json_transcripcion={}
    for i,segment in enumerate(transcripcion):
        json_transcripcion[i]={
            "start": segment[0],
            "end": segment[1],
            "speaker": segment[2],
            "text": segment[3]
        }
    return json_transcripcion

def display_transcripcion(transcripcion):
    texto=""
    if transcripcion is not None:
        for i,segment in transcripcion.items():
            texto+=f"<strong>{i+1} </strong>{segment['start']} -- {segment['end']} {segment['speaker']}: {segment['text']}<br>"
        st.markdown(texto, unsafe_allow_html=True)



def get_scene_transcription(transcription,tiempo_inicio,tiempo_fin):
    transcripcion=[]
    tini=datetime.datetime.strptime(tiempo_inicio, '%H:%M:%S.%f')
    tfin=datetime.datetime.strptime(tiempo_fin, '%H:%M:%S.%f')
    for i,segment in transcription.items():
        segini=datetime.datetime.strptime(segment['start'], '%H:%M:%S.%f')
        segfin=datetime.datetime.strptime(segment['end'], '%H:%M:%S.%f')
        if segini<=tini and segfin>=tini:
            transcripcion.append([segment['start'], segment['end'], segment['speaker'], segment['text']])
        if segini>=tini and segfin<=tfin:
            transcripcion.append([segment['start'], segment['end'], segment['speaker'], segment['text']])
    if len(transcripcion)==0:
        transcripcion=[tiempo_inicio, tiempo_fin, '<S#0>', "No se ha detectado ninguna voz en la grabación"]
    return str(transcripcion)
        
def get_speakerandtexto(transcription):
    print("get_speakerandtexto")
    print(transcription)
    print(type(transcription))
    transcrip=[]
    if len(transcription)>0:
        for segment in literal_eval(transcription):
            print(segment)
            transcrip.append([segment[2],segment[3]])
    else:
        transcrip=['<S#>', "No se ha detectado ninguna voz en la grabación"]
    return str(transcrip)

def promptfijo(transcripcion_escena, historia):
    texto=f"""
    Eres un modelo de lenguaje visual especializado en identificar contenido que pueda reflejar o promover formas de discriminación y 
    odio en las imágenes, vídeos y transcripciones de audio.

    La transcripción del audio del plano de montaje que vamos a analizar es la siguiente: {transcripcion_escena}.

    El formato de la transcripción es json donde 'start' y 'end' son los tiempos de inicio y fin de la transcripción, 'speaker' 
    es el identificador del hablante y 'texto' es la transcripción del hablante.
    """    
    # Además, en los planos de montaje anteriores has justificado tu decisión en los siguientes hechos: 
    
    # {historia}.
    
    # Ayúdate de esta información para los casos en los que para el plano que estás analizando no queda claro si hay alguna forma de discriminación u ódio.
    # Si el análisis del plano actual es clara la forma de discriminación u odio, ya sea en el vídeo o en el audio o en ambos, puedes tomar la decisión directamente sin utilizar los hechos anteriores.

    return texto

def prompteditable():

    texto="""

    La definición de las formas de discriminación y odio son las siguiente:
    
    Violencia física: Actos de agresión física, amenazas de violencia, imágenes de lesiones o daños físicos, o contenido que glorifique la violencia.

    LGBTQ+fobia (Homofobia, Transfobia, Bifobia): Contenido que denigre, insulte, discrimine o promueva prejuicios contra personas lesbianas, gays, bisexuales, transgénero o de otras identidades de género y orientaciones sexuales. Incluye discursos de odio, insultos, negación de derechos y promoción de la violencia contra personas LGBTQ+.

    Machismo (Sexismo, Misoginia): Contenido que perpetúa estereotipos de género dañinos, discrimina o devalúa a las mujeres, o promueve la superioridad masculina. Incluye insultos sexistas, objetificación, acoso, y la promoción de la violencia contra las mujeres.

    Racismo y Xenofobia: Contenido que discrimina, insulta o promueve prejuicios contra personas por su raza, origen étnico, nacionalidad o ascendencia. Incluye estereotipos raciales, insultos racistas, la promoción de la discriminación y la incitación al odio racial.

    Antigitanismo: Contenido que discrimina, insulta o promueve prejuicios contra el pueblo gitano. Incluye estereotipos negativos, insultos, la negación de sus derechos y la promoción de la exclusión social.

    Islamofobia: Contenido que discrimina, insulta o promueve prejuicios contra personas musulmanas o contra el Islam en general. Incluye estereotipos negativos, insultos, la promoción del odio y la justificación de la discriminación.

    Antisemitismo: Contenido que discrimina, insulta o promueve prejuicios contra personas judías. Incluye estereotipos negativos, insultos, la negación del Holocausto, teorías de conspiración y la promoción del odio.

    Aspectismo: prejuicio o discriminación hacia las personas que se consideran físicamente poco atractivas.
    

    Utiliza siempre el siguiente formato para tu respuesta:

    {
    "justificacion": "texto con la justificación.",
    "clasificacion": "Clasificación del vídeo según las categorías definidas."
    }
    
    Si tienes que escribir comillas dobles en tu respuesta, utiliza siempre comillas simples para delimitar la cadena de texto.
    Recuerda en la descripción de la escena utiliza simpre comillas simples, nunca comillas dobles. Está prohibido el uso de comillas dobles en la descripción de la escena.
    En ningún caso utilices comillas dobles en la descripción de la escena.
    Indicar si se detecta en las imágenes o en el audio odio, violencia, lenguaje ofensivo o contenido inapropiado.
    Proporciona la descripción de la escena con detalle pero escribe solo en español.

    ###Instrucciones para realizar el análisis:

    Análisis Contextual: Evalúa la imagen en su contexto completo. Considera el texto que la acompaña, la fuente de la imagen, el propósito aparente de la comunicación y la descripción de los planos anteriores.

    Identificación de Indicadores: Busca indicadores visuales y textuales que puedan sugerir la presencia de alguna de las formas de discriminación y odio mencionadas. Ten encuenta los indicadores previos.

    Priorizar la Intención: Intenta determinar la intención del creador del contenido. ¿Está tratando de promover el odio o la discriminación, o está simplemente informando sobre un problema o expresando una opinión controversial?

    Evitar Falsos Positivos: Sé cauteloso al identificar contenido como discriminatorio u odioso. Asegúrate de que haya evidencia clara y convincente antes de hacer una clasificación.

    Sensibilidad Cultural: Ten en cuenta las diferencias culturales y las posibles interpretaciones erróneas del contenido.

    Manejo de la Ambigüedad: Si no estás seguro de si una imagen es discriminatoria u odiosa, clasifícala como "incierta" o "necesita revisión humana".

    Sesgos: Presta especial atención a los sesgos que puedes tener.

    Clasificación: Clasifica en una o más de las siguientes categorías:

        Violencia Física

        LGBTQ+fobia

        Machismo

        Racismo y Xenofobia

        Antigitanismo

        Islamofobia

        Antisemitismo

        Aspectismo

        Incerteza/Necesita Revisión Humana

        No contiene contenido inapropiado

    ###Ejemplo de Salida:
    {
    'clasificacion': 'Racismo y Xenofobia, Machismo',
    'justificacion': 'El video muestra [descripción del vídeo y transcripción de audio] y contiene elementos que perpetúan estereotipos raciales y sexistas al 
    [explicación de cómo se perpetúan los estereotipos]'.
     """
    return texto

def prompt_history(history):
    texto=f"""
    Esta es la justificación de los dos planos anteriores:
    {history}
    
    Utiliza esta información para tomar una decisión en el plano actual.
    """
    return texto


def prompt(transcription_escena,terminos,tiempo_inicio="00:00:00",tiempo_fin="00:00:00"):
# Prompt text
# La transcripción del audio del vídeo completo es la siguiente: """+transcription+""" 
    texto ="""La siguiente secuencia de frames se corresponde con una escena de un vídeo que empieza en el tiempo """+tiempo_inicio+""" 
    y termina en el tiempo """+tiempo_fin+""".<br>
    Tu función es clasificar la escena en términos de si contiene contenido inapropiado.<br>
    Se trata de detectar si en el vídeo o en la transcripción del audio hay indicios de <strong>"""+terminos+"""</strong>.<br>
    Escribe solo en español.<br>
    Recuerda que la descripción de la escena debe ser objetiva y precisa y en términos de detectar si hay <strong>"""+terminos+"""</strong>.<br>
    La transcripción del audio de la escena es la siguiente: """+transcription_escena+""".<br>
    El formato de la transcripción es json donde 'start' y 'end' son los tiempos de inicio y fin de la transcripción, 'speaker' 
    es el identificador del hablante y 'texto' es la transcripción del hablante.<br>
    Utiliza el siguiente formato para tu respuesta:<br>
    {
    "justificacion": "Descripción de la escena en terminos de si hay <strong>"""+terminos+"""</strong>",
    "clasificacion": "Clase de la escena: """+terminos+""" o no contiene contenido inapropiado."
    }
    Si tienes que escribir comillas dobles en tu respuesta, utiliza siempre comillas simples para delimitar la cadena de texto.
    Recuerda en la descripción de la escena utiliza simpre comillas simples, nunca comillas dobles. Está prohibido el uso de comillas dobles en la descripción de la escena.
    En ningún caso utilices comillas dobles en la descripción de la escena.
    Indicar si se detecta en las imágenes o en el audio odio, violencia, lenguaje ofensivo o contenido inapropiado.
    Proporciona la descripción de la escena con detalle pero escribe solo en español.
    
    """
    return texto

def clear_all():
    st.session_state.uploaded_file_state = None
    st.session_state.video_url = ""
    st.session_state.description = ""
    st.session_state.transcripcion = None
    st.session_state.scene_list = {}
    st.session_state.scene_description = {}
    st.session_state.scenes = {}
    st.session_state.instrucciones= prompteditable()
    st.session_state.historia=[]

def visualizar_escena(scene,i,scenefilename):
    if 'no contiene contenido inapropiado' in scene['clasificacion'].lower():
        icon = "🟢"
    else:
        icon = "🔴"
    with st.expander(f"Plano {i+1}: {scene['clasificacion']}",icon=icon):
        scol1, scol2 = st.columns([0.3, 0.7])
        with scol1:
            try:
                st.video(scenefilename,format="video/mp4")
            except:
                st.write('short video')
        with scol2:
            text=""
            for key in scene.keys():
                if key != "transcripcion":
                    text+=f"<strong>{key}:</strong> {scene[key]}<br>"
            st.markdown(text, unsafe_allow_html=True)
        st.write("Transcripción de la escena")
        st.write(scene['transcripcion'])    

def visualizar_escenas(scenes,filename):
    """
    Visualiza las escenas detectadas en el vídeo.
    Utiliza el método st.expander para mostrar cada escena detectada.
    presentando el tiempo de inicio y fin de
    cada escena, la descripción de la escena y la transcripción del audio de la
    escena.
    muestra el vídeo de la escena.
    Args:
        scenes (dict): Diccionario con las escenas detectadas en el vídeo.
    
    Returns:
        None

    """
    for i, scene in scenes.items():
        scenefilename=f"escenas/{filename}-Scene-{i+1:03d}.mp4" 
        visualizar_escena(scene,i,scenefilename)

def obtener_nombre_archivo_de_url(url):
    """
    Extrae el nombre del archivo de una URL.

    Args:
        url: La URL del archivo.

    Returns:
        El nombre del archivo, o None si no se puede extraer.
    """
    try:
        # Analizar la URL usando urllib.parse
        parsed_url = urlparse(url)

        # Obtener la ruta de la URL
        path = parsed_url.path

        # Extraer el nombre del archivo de la ruta usando os.path.basename
        nombre_archivo = os.path.basename(path)

        # Si el nombre del archivo está vacío (por ejemplo, la URL termina en /),
        # intenta obtenerlo de los headers Content-Disposition (si están presentes)
        if not nombre_archivo:
            respuesta = requests.head(url)  # Usar HEAD para no descargar el contenido
            respuesta.raise_for_status()

            content_disposition = respuesta.headers.get("Content-Disposition")
            if content_disposition:
                # Buscar el nombre del archivo en el header Content-Disposition
                for part in content_disposition.split(";"):
                    part = part.strip()
                    if part.startswith("filename="):
                        nombre_archivo = part.split("=", 1)[1].strip('"') # remover quotes do nome do arquivo se houver
                        break

        # Devolver el nombre del archivo
        return nombre_archivo

    except requests.exceptions.RequestException:
        print("Error al obtener las cabeceras de la URL.")
        return None
    except Exception as e:
        print(f"Error al extraer el nombre del archivo: {e}")
        return None
    
def descargar_video(url, nombre_archivo):
  """
  Descarga un video MP4 desde una URL y lo guarda en un archivo local.

  Args:
    url: La URL del video MP4.
    nombre_archivo: El nombre del archivo donde se guardará el video.
  """

  try:
    # Realizar una petición GET a la URL del video con streaming
    respuesta = requests.get(url, stream=True)

    # Verificar si la petición fue exitosa (código 200)
    respuesta.raise_for_status()  # Lanza una excepción en caso de error

    # Abrir el archivo en modo binario para escritura
    with open(nombre_archivo, 'wb') as archivo:
      # Iterar sobre los chunks de la respuesta y escribirlos en el archivo
      for chunk in respuesta.iter_content(chunk_size=8192):  # 8KB por chunk
        if chunk:  # Filtrar chunks vacíos
          archivo.write(chunk)

    print(f"Video descargado exitosamente a: {nombre_archivo}")

  except requests.exceptions.RequestException as e:
    print(f"Error al descargar el video: {e}")
  except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")

def procesar_audio(videofilename):
    with st.spinner("Transcribiendo el vídeo ..."):
        transcripcion_video=transcribe_video(videofilename)
        st.session_state.transcripcion=transcripcion_video
    # descargamos la transcripción del video en formato csv 

def main():


    st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
    
    if os.path.exists("Cabecera_Cemiya.jpg"):
        st.image("Cabecera_Cemiya.jpg", width=400)
    else:
        st.warning("Imagen 'Cabecera_Cemiya.jpg' no encontrada. Verifica que esté en la carpeta del proyecto.")

    with st.sidebar:
        st.image("MICIUCofinanciadoAEI-768x149.jpg",width=400)
        with st.expander("Información de uso"):
            st.markdown("""
                    ## Configuración del sistema de análisis:

                    1.  Se puede seleccionar el idioma del audio y el modelo de transcripción a utilizar.
                    2.  Se puede seleccionar el modelo de lenguaje visual a utilizar.
                    3.  Se puede particularizar las instrucciones para el análisis del vídeo.

                    ## Funcionamiento:

                    1.  Cargar un video con "Browse file" o escribiendo la url donde se encuentra el video mp4.
                    2.  Comprobar que las instrucciones son las correctas para la tarea. ##No modificar el formato de salida.## Se puede modificar las categorias y su descripción. Para que se actualizar terminar con Ctrl+Enter.
                    2.  Una vez cargado se puede:
                        *   2.a. Transcribir el audio pulsando el botón "transcribir el vídeo" que aparece debajo del contenedor de visualización del vídeo.
                        *   2.b. Dividir el vídeo es planos de montaje pulsando el botón "Detectar planos de montaje".
                    3.  Una vez dividido el vídeo en planos de montaje se puede analizar pulsando el botón "Analizar vídeo por planos".

                    Tanto la transcripción como el análisis por planos se puede descargar en formato excel pulsando el respectivo botón.
                     """)
        with st.container(border=True):
            st.write("Configuración audio")
            parametros['language'] = st.selectbox('Selecciona el idioma ', supportedLanguages,index=0)
            st.markdown("""---""")
            parametros['model_size'] = st.selectbox('Selecciona el tamaño del modelo (def: large-v3)', ['small', 'medium', 'large-v3'], index=2)
        with st.container(border=True):
            st.write("Configuración modelo de lenguaje")
            vlmodel = st.selectbox('Selecciona el modelo', ['Qwen/Qwen2.5-VL-7B-Instruct','Qwen/Qwen2.5-VL-3B-Instruct'], index=0)
        # with st.container(border=True):
        #     st.write("Términos a detectar")
        #     terminos = st.text_area("Escribe los términos a detectar en el vídeo", terminos_base,height=200)


    if "file_uploader" not in st.session_state:
        st.session_state.uploaded_file_state = None
    if "video_url" not in st.session_state:
        st.session_state.image_url = ""
    if "description" not in st.session_state:
        st.session_state.description = ""
    if "transcripcion" not in st.session_state:
        st.session_state.transcripcion = None
    if "scene_list" not in st.session_state:
        st.session_state.scene_list = {}
    if "scene_description" not in st.session_state:
        st.session_state.scene_description = {}
    if "scenes" not in st.session_state:
        st.session_state.scenes = {}
    if "instrucciones" not in st.session_state:
        st.session_state.instrucciones= prompteditable()
    if "historia" not in st.session_state:
        st.session_state.historia=[]
        
    # Load the model and processor
#    model, processor = load_qwenmodel()
    client = qwen_client()
    videourl = None
    scene_list={}
    scenes=None
    scene_description=None
    transcripcion_video=None
    instrucciones=st.session_state.instrucciones

    if not os.path.exists("videos"):
        os.mkdir("videos")
        
    if not os.path.exists("escenas"):
        os.mkdir("escenas")
        
    col_1, col_2 = st.columns([0.7,0.3])
    with col_1.container():
        uploaded_file = st.file_uploader(
            "Sube un vídeo para analizar", type=["mp4"], key="file_uploader",on_change=clear_all
        )
        st.text_input("O pega aquí la url del vídeo", key="video_url")#,on_change=clear_text)
        videourl = st.session_state.video_url
        with st.expander("Instrucciones"):
            instrucciones=st.text_area(" ",st.session_state.instrucciones,height=65*10)
            st.session_state.instrucciones=instrucciones
#        st.expander("Instrucciones").markdown(prompt('Aquí va la transcrición de la escena',terminos,"00:00:00","00:00:00"),unsafe_allow_html=True)
        
    with col_2.container(border=True):
        if uploaded_file is not None:
            st.session_state.uploaded_file_state = uploaded_file.getvalue()
            # get file name extension from uploaded file
            ext = uploaded_file.name.split(".")[-1]
            filename=uploaded_file.name
            filename=filename.split(".")[0]
            videofilename="videos/"+filename+".mp4"
            # transcribe the video

            if ext in ["mp4"]:
                st.video(uploaded_file,format="video/mp4")
            
                # save the video to a file
                with open(videofilename, "wb") as f:
                    f.write(uploaded_file.getvalue())                

        if videourl:
            filename=obtener_nombre_archivo_de_url(videourl)
            videofilename="videos/"+filename+".mp4"
            # transcribe the video            
            st.video(videourl,format="video/mp4")
            descargar_video(videourl,videofilename)
            
       
        
    col1,col2=st.columns(2)
    if uploaded_file is not None or videourl:   
        with col1.container(border=True):
            col11,col12=st.columns(2)
            with col11:
                if st.button("1. Transcribir el vídeo",type="primary",icon=":material/transcribe:"):
                    procesar_audio(videofilename)
            with col12:
                if st.session_state.transcripcion:
            
                    transcriptionfilename=filename+"-transcripcion.xlsx"
                    st.download_button(
                        "Descargar transcripción del vídeo",
                        dict_to_excel(st.session_state.transcripcion),
                        file_name=transcriptionfilename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        icon=":material/download:",
                    )
                    
            with st.expander("Transcripción del video"):
                display_transcripcion(st.session_state.transcripcion)  
            # scene detection
    
        with col2.container(border=True):

            if st.button("2. Detectar planos de montaje",type="primary",icon=":material/scene:"):
                with st.spinner("Detectando planos ..."):
                    scene_list = split_video(videofilename,filename)
                    st.session_state.scene_list = scene_list
            st.write("Número de planos de montaje detectados:", len(st.session_state.scene_list))

    if st.session_state.scene_list:
        with st.container(border=True):
            col3,col4=st.columns(2)
            with col3:
                if st.session_state.scene_list and st.button("3. Analizar vídeo por planos",type="primary",icon=":material/analytics:"):
                    if not st.session_state.transcripcion:
                        st.write("Se analizará el vídeo sin transcripción")
                        
                    st.session_state.scene_description={}
                    for i, scene in enumerate(st.session_state.scene_list):     
                        # get the scene video from the scenes folder. The scene number has 3 digits
                        scene_video = f"escenas/{filename}-Scene-{i+1:03d}.mp4"  
                        print(scene_video)      

                        # check if the scene video exists
                        if os.path.exists(scene_video):
                            # transcribe the video
        #                        transcription=get_speakerandtexto(transcribe_video(scene_video))
        #                        print(transcription)
                            if st.session_state.transcripcion:
                                transcription=get_scene_transcription(st.session_state.transcripcion,scene[0].get_timecode(),scene[1].get_timecode())
                            else:
                                transcription="No se ha detectado ninguna voz en la grabación"
        #                        print(transcription) 
                            # get scene duration in frames
                            textoprompt=promptfijo(transcription, st.session_state.historia)+st.session_state.instrucciones
                            print("Prompt.........................")
                            print(promptfijo(transcription, st.session_state.historia))
                            print(".................................")
                            scene_duration = scene[1].get_frames() - scene[0].get_frames()
                            # establish the frame rate for the vllm model
                            if scene_duration < 3000:
                                fps=1
                            elif scene_duration < 4000:
                                fps=0.5
                            else:
                                fps=0.25
                            messages = [{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": 
                                            {
                                            'url': encode_base64_mp4_from_localfile(scene_video)
                                            },
                                        "max_pixels": 360 * 420,
                                        "fps": fps,
                                    },
                                    {"type": "text", "text": textoprompt},
#                                    {"type": "text", "text": prompt(transcription,terminos,scene[0].get_timecode(),scene[1].get_timecode())},
                                ],
                            }]  
                            print("scene duration: ", scene_duration, "fps: ",fps) 
                            with st.spinner(f"Analizando plano {i+1} ..."):                         
                                scene_description = get_description(vlmodel, client, messages)

                            print(scene_description)
                            print(type(scene_description))
                            try:
                                scene_description=json.loads(scene_description)
                            except:
#                                print("Error en el formato dict")
                                scene_description=scene_description
                            #
                            print(scene_description.keys())
                            st.session_state.scene_description[i]=scene_description
                            # create a dict with the scene description fields, timecodes and scene number
                            st.session_state.scenes[i] = {
                                "tiempo_inicio": scene[0].get_timecode(),
                                "tiempo_final": scene[1].get_timecode(),
                                "justificacion": scene_description['justificacion'],
                                "clasificacion": scene_description['clasificacion'],
                                "transcripcion": transcription
                            }
                            # display the scene numbre, timecodes and description.
                            visualizar_escena(st.session_state.scenes[i],i,scene_video)
                            st.session_state.historia.append(scene_description['justificacion'])
                            if len(st.session_state.historia)>2:
                                st.session_state.historia = st.session_state.historia[-2:]
                                                    


        #                            scenes[i+1] = {"description": scene_description, "start_time": scene[0].get_timecode(), "end_time": scene[1].get_timecode(), "transcription": transcription}

                        else:
                            st.write(f"Plano {i+1} not found")
                                            # guardar en un fichero json las escenas
                    st.rerun()
            with col4:
                if st.session_state.scenes:
                    # change scene description to csv format
                    scene_description={}
                    for i,scene in st.session_state.scenes.items():
                        scene_description[i]=scene

                    # filename for the csv file
                    descriptionfilename=filename+"-justificacion.xlsx"
                    st.download_button(
                        label="Descargar descripción de los planos de montaje",
                        data=dict_to_excel(scene_description),
                        file_name=descriptionfilename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        icon=":material/download:",
                    )
                            # Comprobar si el directorio existe
                    if not os.path.exists("justificacion"):
                        # Si no existe, crear el directorio (y los directorios intermedios si es necesario)
                        os.makedirs("justificacion")
                        print(f"Directorio creado")

                    # Save scene description to an Excel file
                    scene_description_excel = dict_to_excel(scene_description)
                    with open(f"justificacion/{descriptionfilename}", "wb") as f:
                        f.write(scene_description_excel)

                    # visualizar las escenas
            visualizar_escenas(st.session_state.scenes,filename)

        # with open("scenes.json", "w") as f:
        #         json.dump(scenes, f)
    # else:    
    #     st.write("No se ha cargado ningún vídeo ni se ha proporcionado ninguna URL")  
        

if __name__ == "__main__":
    main()



