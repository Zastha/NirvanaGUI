import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from datetime import datetime
import cv2
import threading
import time
import subprocess

# Lista de videos
video_list = sorted([f for f in os.listdir("assets") if f.startswith("vid") and f.endswith(".mkv")])
first_video_duration = 10  # Duración en segundos para el primer video
video_duration = 5         # Duración en segundos para los demás videos
neutral_video_path = os.path.join("assets", "neutral.mkv")

# Diccionario para almacenar los datos del usuario
user_data = {
    "Nombre": "",
    "Apellido": "",
    "Nacionalidad": "",
    "Edad": "",
    "Género": "",
    "Textos": [],
    "Video_Data": []
}

# Subproceso para el script de detección facial
facial_process = None

# Función para guardar los datos del usuario en variables temporales
def save_user_data():
    user_data["Nombre"] = entry_nombre.get()
    user_data["Apellido"] = entry_apellido.get()
    user_data["Nacionalidad"] = entry_nacionalidad.get()
    user_data["Edad"] = entry_edad.get()
    user_data["Género"] = gender_var.get()

    if not all([user_data["Nombre"], user_data["Apellido"], user_data["Nacionalidad"], user_data["Edad"], user_data["Género"]]):
        messagebox.showwarning("Input Error", "Todos los campos son obligatorios")
        return

    messagebox.showinfo("Success", "Datos guardados temporalmente")
    
    # Reproducir todos los videos
    threading.Thread(target=play_all_videos).start()

# Función para guardar los datos en un archivo CSV al finalizar el programa
def save_to_csv():
    nombre = user_data["Nombre"]
    # Crear la carpeta si no existe
    carpeta_path = os.path.join("csv", nombre)
    os.makedirs(carpeta_path, exist_ok=True)

    # Generar timestamp y ruta del archivo CSV
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_path = os.path.join(carpeta_path, f"{nombre}_{timestamp}.csv")

    # Datos personales
    personal_data = {
        "Nombre": [user_data["Nombre"]],
        "Apellido": [user_data["Apellido"]],
        "Nacionalidad": [user_data["Nacionalidad"]],
        "Edad": [user_data["Edad"]],
        "Género": [user_data["Género"]]
    }
    df_personal_data = pd.DataFrame(personal_data)
    df_personal_data.to_csv(csv_path, index=False)

    # Datos de videos
    if user_data["Video_Data"]:
        df_video_data = pd.DataFrame(user_data["Video_Data"])
        df_video_data.to_csv(csv_path, mode='a', header=True, index=False)

    messagebox.showinfo("Success", f"Datos guardados en {csv_path}")

# Función para reproducir todos los videos
def play_all_videos():
    for index, video in enumerate(video_list):
        video_path = os.path.join("assets", video)
        if index == 0:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            play_video(video_path, first_video_duration)
            user_data["Video_Data"].append({
                "Video": f"Video {index+1}",
                "Start Time": start_time,
                "End Time": "",
                "sentiment_prompt": ""
            })
        else:
            # Reproducir video neutral
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            play_video(neutral_video_path, video_duration)
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_data["Video_Data"].append({
                "Video": f"Neutral Video {index}",
                "Start Time": start_time,
                "End Time": end_time,
                "sentiment_prompt": ""
            })

            # Reproducir video numerado
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            play_video(video_path, video_duration)
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_data["Video_Data"].append({
                "Video": f"Video {index+1}",
                "Start Time": start_time,
                "End Time": end_time,
                "sentiment_prompt": ""
            })

            # Obtener el texto del usuario después de cada video numerado
            get_user_text(f"Video {index+1}")

# Función para reproducir el video
def play_video(video_path, duration=5):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()

    # Configurar la ventana en pantalla completa
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para obtener el texto del usuario en pantalla completa
def get_user_text(video_name):
    input_window = tk.Toplevel(root)
    input_window.attributes('-fullscreen', True)
    input_window.configure(bg='white')

    # Initialize wait_var here to ensure scope
    wait_var = tk.IntVar()

    def save_input():
        user_text = text_box.get("1.0", tk.END).strip()
        if user_text:
            for video in user_data["Video_Data"]:
                if video["Video"] == video_name:
                    video["sentiment_prompt"] = user_text
                    break
        input_window.destroy()
        wait_var.set(1)

    label = tk.Label(input_window, text="Ingresa tu texto:", font=('Helvetica', 24), bg='white')
    label.pack(pady=20)

    text_box = tk.Text(input_window, font=('Helvetica', 16), wrap='word')
    text_box.pack(expand=True, fill='both', padx=20, pady=20)

    button = tk.Button(input_window, text="Save", font=('Helvetica', 16), command=save_input)
    button.pack(pady=20)

    # Bind the Escape key to close the window and set wait_var
    input_window.bind("<Escape>", lambda e: (input_window.destroy(), wait_var.set(1)))

    # Ensure wait_var is set when the window is closed
    input_window.protocol("WM_DELETE_WINDOW", lambda: (input_window.destroy(), wait_var.set(1)))

    # Wait for the input window to close
    root.wait_variable(wait_var)

# Función para cerrar la aplicación al presionar Escape y guardar los datos
def on_escape(event):
    save_to_csv()
    if facial_process:
        facial_process.terminate()
    root.destroy()

# Función para iniciar el script FacialEmo.py
def start_facial_emotion_script():
    global facial_process
    facial_process = subprocess.Popen(["python", "C:\\Users\\Sarah\\OneDrive\\Desktop\\NirvanaGUI\\FacialEmo.py"])

# Crear la ventana principal
root = tk.Tk()
root.title("Survey")
root.attributes('-fullscreen', True)

# Vincular la tecla Escape a la función on_escape
root.bind("<Escape>", on_escape)

# Iniciar el script de detección facial
start_facial_emotion_script()

# Estilo personalizado
style = ttk.Style()
style.configure('TLabel', font=('Albertus Extra Bold', 16))
style.configure('TButton', font=('Albertus Extra Bold', 16))
style.configure('Header.TLabel', font=('Albertus Extra Bold', 24, 'bold'), foreground='blue')

# Frame para el título
title_frame = ttk.Frame(root)
title_frame.pack(side=tk.TOP, pady=20)
title_label = ttk.Label(title_frame, text="Multimodal Emotion Detection", style='Header.TLabel')
title_label.pack()

# Frame principal
frame = ttk.Frame(root, padding="10")
frame.pack(expand=True)

# Etiquetas y entradas para los datos
label_nombre = ttk.Label(frame, text="Nombre:")
label_nombre.grid(row=0, column=0, padx=10, pady=10)
entry_nombre = ttk.Entry(frame, font=('Helvetica', 16))
entry_nombre.grid(row=0, column=1, padx=10, pady=10)

label_apellido = ttk.Label(frame, text="Apellido:")
label_apellido.grid(row=0, column=2, padx=10, pady=10)
entry_apellido = ttk.Entry(frame, font=('Helvetica', 16))
entry_apellido.grid(row=0, column=3, padx=10, pady=10)

label_nacionalidad = ttk.Label(frame, text="Nacionalidad:")
label_nacionalidad.grid(row=1, column=0, padx=10, pady=10)
entry_nacionalidad = ttk.Entry(frame, font=('Helvetica', 16))
entry_nacionalidad.grid(row=1, column=1, padx=10, pady=10)

label_edad = ttk.Label(frame, text="Edad:")
label_edad.grid(row=2, column=0, padx=10, pady=10)
entry_edad = ttk.Entry(frame, font=('Helvetica', 16))
entry_edad.grid(row=2, column=1, padx=10, pady=10)

# Apartado para género
label_genero = ttk.Label(frame, text="Género:")
label_genero.grid(row=3, column=0, padx=10, pady=10)

gender_var = tk.StringVar()
radiobutton_male = ttk.Radiobutton(frame, text="Hombre", variable=gender_var, value="Hombre")
radiobutton_male.grid(row=3, column=1, padx=10, pady=10, sticky="w")

radiobutton_female = ttk.Radiobutton(frame, text="Mujer", variable=gender_var, value="Mujer")
radiobutton_female.grid(row=3, column=2, padx=10, pady=10, sticky="w")

# Botón para guardar los datos temporalmente
button_next = ttk.Button(frame, text="Next", command=save_user_data)
button_next.grid(row=4, columnspan=4, pady=20)

# Ejecutar el bucle principal
root.mainloop()
