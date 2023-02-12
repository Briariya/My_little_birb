#version 0.0.3
import time
import datetime
import telebot

token = '5734661771:AAExKRzJuvg8dKRWm_RH_CV4JVWSMK0YcNc' # @popugaii_1_bot
bot = telebot.TeleBot(token)
@bot.message_handler(commands=['start'])
def command_send(message):
    text_ = "Покорми попугая"
    bot.send_message(message.chat.id, text_)
    time.sleep(10)
    command_send(message)

# Функция для отправки сообщения
def send_message(message):
    command_send(message)
    pass

# Функция для проверки времени
def check_time():
    # Получаем текущее время
    current_time = datetime.datetime.now()
    # Проверяем, соответствует ли оно указанному времени
    if current_time == datetime.datetime(2023, 2, 11, 12, 28, 0):
        # Отправляем сообщение
        send_message("Время пришло!")

bot.infinity_polling(skip_pending=True)
# Запускаем цикл проверки времени
while True:
    check_time()
    # Задержка в 1 секунду
    time.sleep(10)

