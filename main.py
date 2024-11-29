import os
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar
from threading import Thread
import threading
import queue

def create_folders():
    if not os.path.exists('input'):
        os.makedirs('input')
    if not os.path.exists('output'):
        os.makedirs('output')

def process_videos(selected_filters, increase_fps=False, fps_factor=1):
    input_folder = 'input'
    output_folder = 'output'
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    total_videos = len(video_files)

    if total_videos == 0:
        messagebox.showinfo("Информация", "В папке 'input' нет видеофайлов.")
        enable_widgets()
        return

    # Устанавливаем прогресс-бар для видео
    progress_queue.put(('progress_max', total_videos))
    progress_queue.put(('progress_value', 0))

    disable_widgets()

    for idx, video_file in enumerate(video_files):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)

        try:
            apply_filters_to_video(input_path, output_path, selected_filters, increase_fps, fps_factor)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обработке {video_file}:\n{e}")
            continue

        progress_queue.put(('progress_value', idx + 1))

    message_queue.put("Готово")
    enable_widgets()

def apply_filters_to_video(input_path, output_path, selected_filters, increase_fps, fps_factor):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or original_fps is None or np.isnan(original_fps):
        original_fps = 25  # Значение по умолчанию

    # Новое значение FPS
    if increase_fps:
        fps = original_fps * fps_factor
    else:
        fps = original_fps

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Обновляем прогресс-бар для кадров
    progress_queue.put(('frame_progress_max', frame_count))
    progress_queue.put(('frame_progress_value', 0))

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise IOError(f"Не удалось создать выходной видеофайл {output_path}")

    previous_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Применяем выбранные фильтры
        filtered_frame = apply_selected_filters(frame, selected_filters)

        # Если увеличиваем FPS с помощью дублирования кадров
        if increase_fps:
            out.write(filtered_frame)
            # Дублирование кадра fps_factor - 1 раз
            for _ in range(int(fps_factor) - 1):
                # Можно добавить интерполяцию между кадрами здесь
                out.write(filtered_frame)
        else:
            out.write(filtered_frame)

        current_frame += 1

        # Обновляем прогресс-бар для кадров
        progress_queue.put(('frame_progress_value', current_frame))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def apply_selected_filters(frame, selected_filters):
    result = frame.copy()

    if 'denoise' in selected_filters:
        # Преобразуем изображение в формат подходящий для фильтра
        result = cv2.fastNlMeansDenoisingColored(result, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    if 'sharpen' in selected_filters:
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        result = cv2.filter2D(result, -1, kernel)

    if 'brightness_contrast' in selected_filters:
        alpha = 1.2  # Контрастность (1.0-3.0)
        beta = 0    # Яркость (0-100)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    if 'grayscale' in selected_filters:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if 'sepia' in selected_filters:
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        result = cv2.transform(result, kernel)
        result = cv2.convertScaleAbs(result)

    if 'gaussian_blur' in selected_filters:
        result = cv2.GaussianBlur(result, (5, 5), 0)

    if 'edge_detection' in selected_filters:
        edges = cv2.Canny(result, 100, 200)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if 'histogram_equalization' in selected_filters:
        img_yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    if 'cartoon' in selected_filters:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(result, 9, 250, 250)
        result = cv2.bitwise_and(color, color, mask=edges)

    if 'negative' in selected_filters:
        result = cv2.bitwise_not(result)

    if 'emboss' in selected_filters:
        kernel = np.array([[ -2, -1,  0],
                           [ -1,  1,  1],
                           [  0,  1,  2]])
        result = cv2.filter2D(result, -1, kernel) + 128

    if 'overexposure_correction' in selected_filters:
        # Коррекция переэкспонированных участков с помощью CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result

def start_processing():
    selected_filters = []
    if denoise_var.get():
        selected_filters.append('denoise')
    if sharpen_var.get():
        selected_filters.append('sharpen')
    if brightness_contrast_var.get():
        selected_filters.append('brightness_contrast')
    if grayscale_var.get():
        selected_filters.append('grayscale')
    if sepia_var.get():
        selected_filters.append('sepia')
    if gaussian_blur_var.get():
        selected_filters.append('gaussian_blur')
    if edge_detection_var.get():
        selected_filters.append('edge_detection')
    if histogram_equalization_var.get():
        selected_filters.append('histogram_equalization')
    if cartoon_var.get():
        selected_filters.append('cartoon')
    if negative_var.get():
        selected_filters.append('negative')
    if emboss_var.get():
        selected_filters.append('emboss')
    if overexposure_correction_var.get():
        selected_filters.append('overexposure_correction')

    if not selected_filters and not increase_fps_var.get():
        messagebox.showerror("Ошибка", "Выберите хотя бы один фильтр или увеличение FPS.")
        return

    increase_fps = increase_fps_var.get()
    if increase_fps:
        try:
            fps_factor = float(fps_entry.get())
            if fps_factor <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное значение для увеличения FPS (больше 1).")
            return
    else:
        fps_factor = 1  # Значение по умолчанию

    # Создаем очереди для обмена данными между потоками
    global progress_queue, message_queue
    progress_queue = queue.Queue()
    message_queue = queue.Queue()

    # Запускаем обработку в отдельном потоке, чтобы не замораживать GUI
    thread = Thread(target=process_videos, args=(selected_filters, increase_fps, fps_factor))
    thread.start()

    # Запускаем функцию обновления интерфейса
    root.after(100, check_queues)

def check_queues():
    try:
        while True:
            item = progress_queue.get_nowait()
            if item[0] == 'progress_max':
                progress['maximum'] = item[1]
            elif item[0] == 'progress_value':
                progress['value'] = item[1]
            elif item[0] == 'frame_progress_max':
                frame_progress['maximum'] = item[1]
            elif item[0] == 'frame_progress_value':
                frame_progress['value'] = item[1]
    except queue.Empty:
        pass

    try:
        msg = message_queue.get_nowait()
        if msg == "Готово":
            messagebox.showinfo("Готово", "Обработка видео завершена.")
    except queue.Empty:
        pass

    if threading.active_count() > 1:
        root.after(100, check_queues)
    else:
        enable_widgets()

def reset_program():
    # Сбрасываем все чекбоксы
    denoise_var.set(False)
    sharpen_var.set(False)
    brightness_contrast_var.set(False)
    grayscale_var.set(False)
    sepia_var.set(False)
    gaussian_blur_var.set(False)
    edge_detection_var.set(False)
    histogram_equalization_var.set(False)
    cartoon_var.set(False)
    negative_var.set(False)
    emboss_var.set(False)
    overexposure_correction_var.set(False)
    increase_fps_var.set(False)
    fps_entry.delete(0, END)
    fps_entry.insert(0, "2")  # Значение по умолчанию

    # Сбрасываем прогресс-бары
    progress['value'] = 0
    frame_progress['value'] = 0

    # Делаем кнопку "Начать обработку" активной
    start_button.config(state=NORMAL)

    # Активируем все чекбоксы
    enable_widgets()

def disable_widgets():
    # Отключаем элементы управления во время обработки
    start_button.config(state=DISABLED)
    reset_button.config(state=DISABLED)
    denoise_check.config(state=DISABLED)
    sharpen_check.config(state=DISABLED)
    brightness_contrast_check.config(state=DISABLED)
    grayscale_check.config(state=DISABLED)
    sepia_check.config(state=DISABLED)
    gaussian_blur_check.config(state=DISABLED)
    edge_detection_check.config(state=DISABLED)
    histogram_equalization_check.config(state=DISABLED)
    cartoon_check.config(state=DISABLED)
    negative_check.config(state=DISABLED)
    emboss_check.config(state=DISABLED)
    overexposure_correction_check.config(state=DISABLED)
    increase_fps_check.config(state=DISABLED)
    fps_entry.config(state=DISABLED)

def enable_widgets():
    # Включаем элементы управления после обработки
    start_button.config(state=NORMAL)
    reset_button.config(state=NORMAL)
    denoise_check.config(state=NORMAL)
    sharpen_check.config(state=NORMAL)
    brightness_contrast_check.config(state=NORMAL)
    grayscale_check.config(state=NORMAL)
    sepia_check.config(state=NORMAL)
    gaussian_blur_check.config(state=NORMAL)
    edge_detection_check.config(state=NORMAL)
    histogram_equalization_check.config(state=NORMAL)
    cartoon_check.config(state=NORMAL)
    negative_check.config(state=NORMAL)
    emboss_check.config(state=NORMAL)
    overexposure_correction_check.config(state=NORMAL)
    increase_fps_check.config(state=NORMAL)
    fps_entry.config(state=NORMAL)

if __name__ == "__main__":
    create_folders()

    # Создаем графический интерфейс
    root = Tk()
    root.title("Видео Фильтры")
    root.geometry("400x800")
    root.resizable(False, False)

    Label(root, text="Выберите фильтры для применения:").pack(pady=10)

    # Переменные для чекбоксов
    denoise_var = BooleanVar()
    sharpen_var = BooleanVar()
    brightness_contrast_var = BooleanVar()
    grayscale_var = BooleanVar()
    sepia_var = BooleanVar()
    gaussian_blur_var = BooleanVar()
    edge_detection_var = BooleanVar()
    histogram_equalization_var = BooleanVar()
    cartoon_var = BooleanVar()
    negative_var = BooleanVar()
    emboss_var = BooleanVar()
    overexposure_correction_var = BooleanVar()
    increase_fps_var = BooleanVar()

    # Чекбоксы для выбора фильтров
    denoise_check = Checkbutton(root, text="Удаление шума", variable=denoise_var)
    denoise_check.pack(anchor=W)

    sharpen_check = Checkbutton(root, text="Повышение резкости", variable=sharpen_var)
    sharpen_check.pack(anchor=W)

    brightness_contrast_check = Checkbutton(root, text="Яркость и контрастность", variable=brightness_contrast_var)
    brightness_contrast_check.pack(anchor=W)

    grayscale_check = Checkbutton(root, text="Черно-белый", variable=grayscale_var)
    grayscale_check.pack(anchor=W)

    sepia_check = Checkbutton(root, text="Сепия", variable=sepia_var)
    sepia_check.pack(anchor=W)

    gaussian_blur_check = Checkbutton(root, text="Размытие Гаусса", variable=gaussian_blur_var)
    gaussian_blur_check.pack(anchor=W)

    edge_detection_check = Checkbutton(root, text="Обнаружение границ", variable=edge_detection_var)
    edge_detection_check.pack(anchor=W)

    histogram_equalization_check = Checkbutton(root, text="Выравнивание гистограммы", variable=histogram_equalization_var)
    histogram_equalization_check.pack(anchor=W)

    cartoon_check = Checkbutton(root, text="Эффект мультфильма", variable=cartoon_var)
    cartoon_check.pack(anchor=W)

    negative_check = Checkbutton(root, text="Негатив", variable=negative_var)
    negative_check.pack(anchor=W)

    emboss_check = Checkbutton(root, text="Тиснение", variable=emboss_var)
    emboss_check.pack(anchor=W)

    overexposure_correction_check = Checkbutton(root, text="Коррекция засветок", variable=overexposure_correction_var)
    overexposure_correction_check.pack(anchor=W)

    # Чекбокс и поле для увеличения FPS
    increase_fps_frame = Frame(root)
    increase_fps_check = Checkbutton(increase_fps_frame, text="Увеличить FPS", variable=increase_fps_var)
    increase_fps_check.pack(side=LEFT)
    Label(increase_fps_frame, text="Коэффициент увеличения:").pack(side=LEFT, padx=5)
    fps_entry = Entry(increase_fps_frame, width=5)
    fps_entry.insert(0, "2")  # Значение по умолчанию
    fps_entry.pack(side=LEFT)
    increase_fps_frame.pack(anchor=W, pady=10)

    # Прогресс-бар для видео
    Label(root, text="Прогресс обработки видео:").pack(pady=5)
    progress = Progressbar(root, orient=HORIZONTAL, length=300, mode='determinate')
    progress.pack(pady=5)

    # Прогресс-бар для кадров
    Label(root, text="Прогресс обработки кадров:").pack(pady=5)
    frame_progress = Progressbar(root, orient=HORIZONTAL, length=300, mode='determinate')
    frame_progress.pack(pady=5)

    # Кнопки управления
    buttons_frame = Frame(root)
    buttons_frame.pack(pady=10)

    start_button = Button(buttons_frame, text="Начать обработку", command=start_processing)
    start_button.pack(side=LEFT, padx=5)

    reset_button = Button(buttons_frame, text="Сбросить", command=reset_program)
    reset_button.pack(side=LEFT, padx=5)

    Label(root, text="Поместите видеофайлы в формате .mp4 в папку 'input'.\nОбработанные файлы будут сохранены в папке 'output'.", wraplength=350, justify=CENTER).pack(pady=10)

    root.mainloop()