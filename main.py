#version 0.0.2
import time
import datetime

# Функция для отправки сообщения
def send_message(message):
    # Код для отправки сообщения
    pass

# Функция для проверки времени
def check_time():
    # Получаем текущее время
    current_time = datetime.datetime.now()
    # Проверяем, соответствует ли оно указанному времени
    if current_time == datetime.datetime(2023, 2, 11, 12, 28, 0):
        # Отправляем сообщение
        send_message("Время пришло!")

# Запускаем цикл проверки времени
while True:
    check_time()
    # Задержка в 1 секунду
    time.sleep(1)
