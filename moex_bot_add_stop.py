
import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import List, Optional
import torch
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message
from aiohttp import ClientSession

import aiohttp
import aiomoex
import talib

import pandas as pd
import numpy as np

import os.path
from backtest.Trader.Trader import Trader
from backtest.TraderRLModels import ActorCriticConv1D, GetStateIndCombConv1D
from backtest.TraderUtil import readSettings, ScaleValues

from backtest.TraderUtil import  RLSettings
from backtest.createDirectory import makeFolderName, makeDir
from backtest.Trader.TraderTelegram import Trader
from config import TOKEN, ADMIN_ID, CHANNEL_ID

INTERVAL=10
if INTERVAL == 10:
    RESAMPLE_NUM = 30
    RESAMPLE_INT = str(RESAMPLE_NUM) + 'T'
    CANDLES_NUM = 14*12 + 3*3
elif INTERVAL == 1:
    RESAMPLE_NUM = 5
    RESAMPLE_INT = str(RESAMPLE_NUM) + 'T'
    CANDLES_NUM = 14*5 + 3*3

class MM:
    MAX = 0
    MIN = 0

class Feauteres:
    names = 0
    minmax = 0

def getFeautersDataFromFile(pathToFile, feauter):
    df = pd.read_csv(pathToFile).set_index("NameOfParam")
    #("df getFeautersDataFromFile(pathToFile, feauter)", df)
    returns_MAX = df.loc["returns_MAX","Values"]
    returns_MIN = df.loc["returns_MIN","Values"]
    RSI_5_MAX = df.loc["RSI_5_MAX","Values"]
    RSI_5_MIN = df.loc["RSI_5_MIN","Values"]
    AD5_returns_MAX = df.loc["AD5_returns_MAX","Values"]
    AD5_returns_MIN = df.loc["AD5_returns_MIN","Values"]
    OneVals = MM();OneVals.MAX = returns_MAX; OneVals.MIN = returns_MIN
    SecondVals = MM();SecondVals.MAX = RSI_5_MAX; SecondVals.MIN = RSI_5_MIN;
    ThreeVals = MM(); ThreeVals.MAX = AD5_returns_MAX; ThreeVals.MIN = AD5_returns_MIN;
    names = feauter
    minmax = [OneVals, SecondVals, ThreeVals]
    feauteresData = Feauteres()
    feauteresData.names = names
    feauteresData.minmax = minmax
    return feauteresData

def logData(file, string):
    with open(file, "a") as myfile:
        #print(file)
        myfile.write(string)

def calculate_rsi(data, rsi_period):
    dataset = data.copy()
    delta = dataset['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    dataset['rsi_df'] = rsi
    dataset['rsi_df'] = dataset['rsi_df'].shift(1)
    return dataset

#TICKER = 'EURUSD'
pathToFile = "/backtest/TraderExamples/Settings/SettingsGAZPTrain_7-10-2024_13-2-2025_01092025.csv"
TICKER, nameOfTicker, timeDelta, folderName, rewardType, nameOfDates, nameOfTestDates , dateName, lots, startCash, comisBroker, comisMoex, del_r, Gamma, Clc, lr, gran, will, rewardMult, willBeGap_using, willBeGap_value, passLoopForward, dontMakeCurrentDeals=readSettings(pathToFile)
PERIOD = '/'+ str(timeDelta) +'m/'
comis = (comisBroker, comisMoex) 

folderName = makeFolderName(folderName, comis[0], Gamma, Clc, lr, del_r, gran, will)
nameOfModel = "GAZP_ActorCriticConv1D_Tr_Rw=410748_Test_Rw=139382_returns_RSI5_sc_AD5_returns_sc__ep=490"

PATHFORMODELS = "/backtest/TrainedModels/" + TICKER + PERIOD + folderName

PATHFORMODELSPTH = PATHFORMODELS+ "pth/" + nameOfModel + ".pth"
PATHFORPDF= PATHFORMODELS + "test/" + nameOfModel
PATHFORJITMODELS = PATHFORMODELS + "l/" + nameOfModel + "_jit.pt"

pathForData = PATHFORMODELS + "data/"
pathForCurrentTestFolder = PATHFORMODELS + "test/" + nameOfModel.split("=")[-1] + "/"


fileLogName = "Log.txt"
isLogExist = os.path.exists(pathForCurrentTestFolder + fileLogName)
if isLogExist:
    fileLogName = "Log2.txt"

nameOfPreparedDataFile = nameOfTicker + nameOfDates + ".csv"

nameOfTestDataFile =  nameOfTicker + nameOfTestDates + "_Test"+ ".csv"

nameOfMinMaxDataFile = nameOfDates + "_MINMAX.csv"

settings = RLSettings(pathForCurrentTestFolder, lots, lots, startCash, comis, rewardType, del_r, gran ,will, timeDelta, rewardMult, Gamma, Clc, lr, willBeGap_using,  willBeGap_value, passLoopForward, dontMakeCurrentDeals)
#########################################################################
#reading dataset
feauteres = ['returns', 'RSI5_sc', 'AD5_returns_sc']
feauteresForTest = ['Close', 'RSI5', 'AD5']
feautersData = getFeautersDataFromFile(pathForData + nameOfMinMaxDataFile, feauteresForTest)
settings.feautData = feautersData
#print("feauteresData.names,  feauteresData.minmax",  feautersData.names,  feautersData.minmax)
#print(feauteres)
numOfFeauteres = len(feauteres)
action_numbers = 3


# # Check its architecture
# model.summary()


# ID чата, куда будут отправляться сообщения о сделках
# Можно получить, написав боту /start и посмотрев в логи
CHAT_ID = None

# --- Роутер для обработки команд ---
router = Router()

def wait_until_next_30min():
    """Ждём до следующего 30-минутного интервала: 00 или 30 минут каждого часа."""
    now = datetime.now()
    # Определяем, сколько секунд до следующей "30" или "00"
    minutes = now.minute
    seconds = now.second + now.microsecond / 1_000_000

    if minutes < 30:
        next_time = now.replace(minute=30, second=0, microsecond=0)
    else:
        next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    wait_seconds = (next_time - now).total_seconds()
    return asyncio.sleep(wait_seconds)

def wait_until_next_interval(interval_minutes: int = 5):
    """
    Ждём до следующего интервала, кратного interval_minutes.
    Например, при interval_minutes=5: 12:00, 12:05, 12:10, 12:15, ...
    """
    now = datetime.now()
    # Сколько минут прошло с начала часа
    minutes_since_hour = now.minute
    # Остаток от деления на интервал
    remainder = minutes_since_hour % interval_minutes
    # Через сколько минут будет следующая "метка"
    if remainder == 0:
        # Уже на границе интервала — ждём до следующего
        next_minute = now.minute + interval_minutes
    else:
        next_minute = (minutes_since_hour - remainder) + interval_minutes

    # Корректируем час, если вышли за пределы часа
    if next_minute >= 60:
        next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)

    wait_seconds = (next_time - now).total_seconds()
    return asyncio.sleep(wait_seconds)

def is_time_to_act(candle_time: datetime, interval_minutes: int = 5) -> bool:
    """
    Проверяет, является ли время свечи "меткой" для действия (например, 12:15, 12:20).
    """
    # Округляем время свечи до ближайшей предыдущей интервальной метки
    minutes = candle_time.minute
    remainder = minutes % interval_minutes
    
    # Если remainder == 0, значит, время кратно интервалу
    return remainder == 0

class Strategy:
    def __init__(
        self,
        settings: RLSettings,
        session: Optional[ClientSession],
        bot: Bot,           # Добавить
        chat_id: int        # Добавить
        
    ):
        self.session = session
        self.settings = settings
        self.check_interval = 10
        self.test = False
        #self.backtest =  Trader(self.settings.startCash, self.settings.comis, True)
        self.backtest = Trader.load_state(money=settings.startCash,
                                        comis=settings.comis,
                                        printBool=True)
        self.bot = bot              # Сохраняем бота
        self.chat_id = chat_id      # Сохраняем chat_id
        self.is_running = False  # Флаг: работает ли стратегия
        self.stop_event = asyncio.Event()  # Событие для остановки
        self.task = None  # Ссылка на задачу
        self.report_task = None



    async def step(self, Signal: int, prices: float):
        endOfDataframe = False
        print("зашли в step")
        current_price_str = f"{prices:.2f}".replace('.', ',')  # 133.21 → 133,21
        ticker_name = "ПАО \"Газпром\""

            # Получаем текущую прибыль
        current_profit = self.backtest.getCurrentProfit(prices)
        if self.backtest.posVolume != 0:
            profit_percent = (current_profit / abs(self.backtest.moneyOnStartDeal)) * 100 if self.backtest.moneyOnStartDeal != 0 else 0
        else:
            profit_percent = 0

        if Signal == 2 and self.backtest.state == "zero" and (not endOfDataframe):# and rsi5 < -rsi_max:
            self.backtest.price_of_pos = prices
            self.backtest.buy(self.settings.lot, prices)
            assert self.backtest.state == "Long", f"Ожидалось состояние 'Long', но получено: {self.backtest.state}"
            
            mess = f"⬆️🟢 Покупка акций {ticker_name} по цене {current_price_str} ₽."
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print(mess)
        elif Signal == 0 and self.backtest.state == "zero" and (not endOfDataframe):# and rsi5 > rsi_max:
            self.backtest.price_of_pos = prices
            self.backtest.sell(self.settings.lot, prices)
            assert self.backtest.state == "Short", f"Ожидалось состояние 'Long', но получено: {self.backtest.state}"
            
            mess = f"⬇️🔴 Продажа акций {ticker_name} по цене {current_price_str} ₽."
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print(mess)
        elif ((Signal == 1 and self.backtest.state == "zero") or (Signal == 1 and self.backtest.state == "Long") or (Signal == 1 and self.backtest.state == "Short") or (Signal == 2 and self.backtest.state == "Long") or (Signal == 0 and self.backtest.state == "Short")) and (not endOfDataframe):
            mess = (
                f"🟡 Ничего не делаем, позиция {self.backtest.state}. "
                f"Цена {current_price_str} ₽."
            )
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print("Ничего")

        elif (self.backtest.state == "Long" and Signal == 0) or (self.backtest.state == "Long" and (Signal == 1 or Signal == 2) and endOfDataframe):
            profit_abs = current_profit
            profit_percent = (profit_abs / self.backtest.moneyOnStartDeal) * 100 if self.backtest.moneyOnStartDeal != 0 else 0
            mess = (
                f"⬇️ Продажа акций {ticker_name} по цене {current_price_str} ₽. "
                f"Профит от сделки {profit_abs:.1f} ₽."
                f"Прибыль сделки {profit_percent:.1f}%."
            )
            self.backtest.price_of_pos = 0
            self.backtest.sell(self.settings.lot, prices)
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print(mess)
            
        elif (self.backtest.state == "Short" and Signal == 2) or (self.backtest.state == "Short" and (Signal == 1 or Signal == 0) and endOfDataframe):
            profit_abs = current_profit
            profit_percent = (profit_abs / self.backtest.start_money) * 100 if self.backtest.start_money != 0 else 0
            mess = (
                f"⬆️ Покупка акций {ticker_name} по цене {current_price_str} ₽. "
                f"Профит от сделки {profit_abs:.1f} ₽."
                f"Прибыль сделки {profit_percent:.1f}%."
            )
            self.backtest.price_of_pos = 0
            self.backtest.buy(self.settings.lot, prices)
            
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print(mess)
            
        elif (self.backtest.state == "zero" and endOfDataframe):# нужно ли это здесь?????????????????
            mess = (
                f"🟡 Ничего не делаем, позиция {self.backtest.state}. "
                f"Цена {current_price_str} ₽."
            )
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print("Ничего 3")
        else:
            mess = "⚪ Ничего не делаем."
            await self.bot.send_message(chat_id=self.chat_id, text=mess)
            await self.bot.send_message(chat_id=CHANNEL_ID, text=mess)
            print("Ничего 4")
    
    def get_daily_profit(self):
        # Чтение файла
        """df = pd.read_csv('out.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print("df", df)
        #df = df[1:]"""
        df = self.backtest.table
        df = df[1:]
        df.index = pd.to_datetime(df.index)
        print("df-++++++++++++++++++++++++++++++++++++++++++++++++++\n", df)
        print("type(df['Date'])", type(df.index), df.index)
        #df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print("df-----------------------------------------------------\n", df)
        # Автоматическое определение последнего дня
        last_date = df.index.date.max()
        print("last_date", last_date)
        last_day = df[df.index.date == last_date]
        print("last_day", last_day)

        # Расчет изменения
        if len(last_day) > 1:
            first_value = last_day['Account_money'].iloc[0]
            last_value = last_day['Account_money'].iloc[-1]
            change = last_value - first_value
            percentage_change = (change / first_value) * 100
            
            print(f"Дата: {last_date}")
            print(f"Начальное значение: {first_value}")
            print(f"Конечное значение: {last_value}")
            print(f"Изменение: {percentage_change:.2f}%")
        else:
            print("Недостаточно данных для расчета за последний день")
        return change, percentage_change


    def _get_observation(self, df):
        """Собираем вектор состояния."""
        # Извлекаем окно свечей
        obj = df.index
        current_step = len(obj)-1
        window_size = 2+1
        feauteresForTest = ['Close', 'RSI5', 'AD5']
        
        #print("извлекаем данные сначала")

        stated = GetStateIndCombConv1D(obj, current_step, df, feauteresForTest, window_size)
        #print("GetStateIndCombConv1D")
        statedSq = torch.squeeze(stated, 0)#torch.Size([2, 4])
        statedScaled = ScaleValues(statedSq, self.settings.feautData, 2, 3)#torch.Size([2, 4])
        print("stated scaled", stated)
        return statedScaled, stated

    async def get_all_candles(self, start, end, interval):
        """Функция получения свечей с MOEX."""
        print("Функция получения свечей с MOEX.")
        async with aiohttp.ClientSession() as session:
            try:
                candles = await aiomoex.get_market_candles(
                            session,
                            security='GAZP',
                            interval=interval,  # 1 - 1 минута
                            start = start,
                            end = end,
                            market='shares',
                            engine='stock'
                        )
                if not candles:
                    print(f"❌ Нет данных с MOEX (попытка { 1})")
                    #await asyncio.sleep(5)
                    #continue
                print("Получили данные.")
                last_candle = candles[-1]
                df_candle = candles[- CANDLES_NUM:]
                df = pd.DataFrame(df_candle)
                df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
                df = df.drop(columns=["begin", "end"])
                df.set_index('datetime', inplace=True)
                #print("df", df, len(df))

                df = df.resample(RESAMPLE_INT).agg({
                    'open': 'first',   # Цена открытия первого периода
                    'high': 'max',     # Максимальная цена за период
                    'low': 'min',      # Минимальная цена за период
                    'close': 'last',   # Цена закрытия последнего периода
                    'volume': 'sum'    # Суммарный объем за период
                })
                df= df.dropna(subset=['open', 'high', 'low', 'close'])
                #print("df30", df)
                df.rename(columns = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume':'Volume'}, inplace=True)
                # Удаляем строки с NaN значениями (могут возникнуть, если за 30 минут не было данных)
                #df_30min.dropna(inplace=True)
                
                #print("30-минутные свечи:")
                #print(df)

                df['Volume'] = df['Volume'].astype(float)

                # Это мои индикаторы
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['returns'] = df['Close'] - df['Close'].shift(1)

                #Это мы начинаем использовать talib
                Close = df.loc[:, "Close"].values
                Open = df.loc[:, "Open"].values
                High = df.loc[:, "High"].values
                Low = df.loc[:, "Low"].values
                Volume = df.loc[:, "Volume"].values
                print("--------------------------------------------------------------")
                #i_rsi5 = talib.RSI(Close, timeperiod=14)#RSI
                rsi_df = calculate_rsi(df, 14)
                df["RSI5"] = rsi_df['rsi_df']

                i_ad5 = talib.AD(High, Low, Close, Volume)
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                #df["RSI5"] = i_rsi5
                df["AD5"] = i_ad5
                last_price = last_candle.get('close')
                print("df*********************************************************\n", df)
                
                return df, last_candle
            except Exception as e:
                    print(f"❌Произошла ошибка при получении свечей: {e}")
        print("❌ Не удалось получить данные с MOEX после 3 попыток")
        return None, None

    
    async def get_last_candle(self, start, end, interval):
        """Функция получения свечей с MOEX."""
        async with aiohttp.ClientSession() as session:
            try:
                candles = await aiomoex.get_market_candles(
                            session,
                            security='GAZP',
                            interval=interval,  # 1 - 1 минута
                            start = start,
                            end = end,
                            market='shares',
                            engine='stock'
                        )
                last_candle_1m = candles[-1]
                print("last_candle_1m", last_candle_1m)
                return  last_candle_1m
            except Exception as e:
                print(f"❌❌Произошла ошибка при получении данных: {e}")
            return None

    
    async def get_securities_info(self):
        """Получить информацию об акции GAZP в режиме TQBR."""
        async with aiohttp.ClientSession() as session:
            try:
                # get_board_securities возвращает информацию о ценных бумагах на доске
                # Используем table='marketdata' для получения текущих рыночных данных
                securities_data = await aiomoex.get_board_securities(
                    session,
                    table='marketdata', # Важно: запрашиваем таблицу marketdata
                    columns=None, # Запрашиваем все колонки
                    board='TQBR',
                    market='shares',
                    engine='stock'
                )

                if securities_data:
                    # Нужно найти запись для GAZP в списке всех бумаг
                    gazp_data = None
                    for item in securities_data:
                        if item.get('SECID') == 'GAZP':
                            gazp_data = item
                            break

                    if gazp_data:
                        #print("\nТекущие рыночные данные для GAZP из get_board_securities (marketdata):")
                        #print(gazp_data)
                        # Поля из вашего предыдущего ответа API
                        last_price = gazp_data.get('LAST')
                        change = gazp_data.get('CHANGE')
                        update_time = gazp_data.get('UPDATETIME')
                        bid = gazp_data.get('BID')
                        offer = gazp_data.get('OFFER')
                        """print(f"\nПоследняя цена (LAST): {last_price}")
                        print(f"Изменение (CHANGE): {change}")
                        print(f"Время обновления (UPDATETIME): {update_time}")
                        print(f"Лучшая цена покупки (BID): {bid}")
                        print(f"Лучшая цена продажи (OFFER): {offer}")"""
                        y = datetime.now()
                        print("datetime.now()", y)
                        # 1. Преобразуем строку времени в объект time
                        time_obj = datetime.strptime(update_time, "%H:%M:%S").time()
                        # 2. Создаём новый datetime: дата из y, время из x
                        new_datetime = datetime.combine(y.date(), time_obj)
                        print(new_datetime)
                        return new_datetime
                    else:
                        print("\nИнформация для GAZP не найдена в данных marketdata.")
                else:
                    print("\nНе удалось получить данные marketdata.")

            except Exception as e:
                print(f"Ошибка при получении информации о ценных бумагах: {e}")
            return None
    
    async def run_episode(self, worker_model):
        """Основной цикл live стратегии."""
        print("Основной цикл live стратегии async def run_episode(worker_model, sessiion)")
        print("подождали")
        last_processed_time = None  # Чтобы не обрабатывать одну свечу дважды
        #is_market_work
        target_time = time(23, 49, 59)
        target_time_wekend = time(19, 00, 00)
        start_time = time(9, 48, 58) 
        while not self.stop_event.is_set():
            try:
                today = datetime.today()
                weekday = today.weekday()
                if weekday == 5:  # Суббота
                    print("Сегодня суббота!")
                elif weekday == 6:  # Воскресенье
                    print("Сегодня воскресенье!")
                else:
                    print("Сегодня будний день")
                day_before_yesterday = today - timedelta(days=5)
                # Форматируем даты
                start = day_before_yesterday.strftime('%Y-%m-%d')
                end = today.strftime('%Y-%m-%d')
                now_time = datetime.now()
                only_time = now_time.time()
                if (only_time > target_time or only_time < start_time) and weekday < 5:
                    print(f"⏳ Время  {only_time}. Торги завершены. Ожидаем начало...")
                    await asyncio.sleep(60)
                    continue
                elif (only_time > target_time_wekend or only_time < start_time) and weekday >= 5:
                    print(f"⏳ Время  {only_time}. Торги завершены. Ожидаем начало...")
                    await asyncio.sleep(60)
                    continue
                print("run_episode", "получаем данные")
                df, last_candle = await self.get_all_candles(start, end, INTERVAL)

                if df is None or last_candle is None:
                    print("💤 Нет данных с MOEX. Ждём 60 секунд...")
                    await asyncio.sleep(60)
                    continue

                if INTERVAL == 10:
                    last_1min_candle = await self.get_securities_info()
                elif INTERVAL == 1:
                    #last_1min_candle = pd.to_datetime(last_candle['end'])
                    last_1min_candle = await self.get_last_candle(end, end, 1)

                if last_1min_candle is None:
                    print("💤 Нет данных с MOEX. Ждём 60 секунд...")
                    await asyncio.sleep(60)
                    continue
                print("getCurrentProfit, ", str(self.backtest.getCurrentProfit(last_candle['close'])))
                print("Цена, ", str(last_candle['close']))
                print("last_candle ", last_candle)
                if len(df) < 2:
                    await asyncio.sleep(10)
                    continue

                # --- 2. Проверяем, завершена ли последняя свеча ---
                last_30min_time = df.index[-1]  # Время 30-минутной свечи
                print("last_30min_time", last_30min_time)
                expected_end = last_30min_time + pd.Timedelta(minutes=RESAMPLE_NUM) - pd.Timedelta(seconds=1)
                print("expected_end ", expected_end)
                if INTERVAL == 10:
                    actual_end = last_1min_candle
                elif INTERVAL == 1:
                    actual_end = pd.to_datetime(last_1min_candle['end'])
                print("actual_end ", actual_end)

                # --- Проверка: свеча завершена? ---
                if actual_end < expected_end:
                    print(f"⏳ Свеча {last_30min_time} ещё не завершена. Ожидаем...")
                    await asyncio.sleep(10)
                    continue
                else:
                    print(f"⏳ Свеча {last_30min_time} завершена. Проверяем, не старая ли она...")

                # --- Проверка: не слишком ли старая свеча? ---
                now = actual_end
                max_delay = pd.Timedelta(seconds=30)  # Максимум 30 секунд после завершения
                time_since_end = now - expected_end
                if time_since_end > max_delay:
                    print(f"⏰ Свеча {last_30min_time} завершена слишком давно ({expected_end}). Пропускаем.")
                    # Все равно обновляем last_processed_time, чтобы не обрабатывать снова
                    if last_processed_time is None or last_30min_time > last_processed_time:
                        last_processed_time = last_30min_time
                    await asyncio.sleep(10)
                    continue
                else:
                    print(f"✅ Свеча {last_30min_time} завершена недавно. Можно обрабатывать.")

                # --- 3. Проверяем, не обработали ли мы уже эту свечу ---
                if last_processed_time is not None and last_30min_time <= last_processed_time:
                    print("last_processed_time is not None and last_30min_time <= last_processed_time")
                    await asyncio.sleep(10)
                    continue
                else:
                    print("Не обработали, идем дальше...")

                # --- 4. Теперь можно действовать ---
                print(f"✅ Свеча {last_30min_time} завершена. Делаем предсказание...")
                obj = df.index
                print("obj", obj)
                profit = self.backtest.quant_money(last_candle['close'], obj[-1])
                print("getCurrentProfit, ", profit)
                print("self.backtest.table", self.backtest.table)

                state, stateF = self._get_observation(df)
                #print("state = self._get_observation(stated)", state)
                with torch.no_grad():
                    policy, value = worker_model(state)
                logits = policy.view(-1)
                action_dist = torch.distributions.Categorical(logits=logits)
                logData("action_020925.csv", str(actual_end)+","+str(last_candle['close'])+","+str(stateF.numpy())+","+str(state.numpy())+","+str(policy.numpy())+"\n")
                action = action_dist.sample().item()  # .item() — int
                print("action", action)
                # await self.ensure_market_open()  # Можно оставить или убрать
                await self.step(action, prices=last_candle['close'])  # получаем исторические данные с MOEX
                # после первого получения исторических данных включаем live режим
                # код стратегии для live
                # сигналы на покупку или продажу обрабатываются в live_check_can_we_open_position
                            # --- 5. Запоминаем, что обработали эту свечу ---
                last_processed_time = last_30min_time

                # --- 6. Ждём следующей возможной свечи ---
                print("Ожидаем следующую 30-минутную свечу...")
                await asyncio.sleep(10)  # Можно увеличить, но не обязательно

            except Exception as e:
                print(f"❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                # На всякий случай — ждём минуту и пробуем снова
                await asyncio.sleep(60)

    async def run_episode2(self, worker_model):
        """Основной цикл — один шаг на завершённой 5-минутной свече."""
        last_processed_time = None  # Чтобы не обрабатывать одну свечу дважды
        expected_end = None
        target_time = time(23, 49, 59)
        target_time_wekend = time(19, 00, 00)
        start_time = time(9, 48, 58) 
        while not self.stop_event.is_set():
            try:
                # --- 1. Получаем данные ---
                today = datetime.today()
                weekday = today.weekday()
                if weekday == 5:  # Суббота
                    print("Сегодня суббота!")
                elif weekday == 6:  # Воскресенье
                    print("Сегодня воскресенье!")
                else:
                    print("Сегодня будний день")
                day_before = today - timedelta(days=3)
                start = day_before.strftime('%Y-%m-%d')
                end = today.strftime('%Y-%m-%d')
                now = datetime.now()
                only_time = now.time()
                if (only_time > target_time or only_time < start_time) and weekday < 5:
                    print(f"⏳ Время  {now}. Торги завершены. Ожидаем начало...")
                    await asyncio.sleep(60)
                    continue
                if (only_time > target_time_wekend or only_time < start_time) and weekday >= 5:
                    print(f"⏳ Время  {now}. Торги завершены. Ожидаем начало...")
                    await asyncio.sleep(60)
                    continue
                df_5min, last_candle = await self.get_all_candles(start, end, INTERVAL)
                if len(df_5min) < 2:
                    await asyncio.sleep(10)
                    continue

                # --- 2. Проверяем, завершена ли последняя свеча ---
                last_5min_time = df_5min.index[-1]  # Время 5-минутной свечи
                print("last_5min_time", last_5min_time)
                last_expected_end = expected_end
                expected_end = last_5min_time + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1)
                print("expected_end", expected_end)
                actual_end = pd.to_datetime(last_candle['end'])
                print("actual_end", actual_end)
                print("last_expected_end", last_expected_end)

                
                if actual_end < expected_end and expected_end == last_expected_end:
                    print(f"⏳ Свеча {last_5min_time} ещё не завершена. Ожидаем...")
                    await asyncio.sleep(10)
                    continue
                elif expected_end != last_expected_end:
                    if last_expected_end is None:
                        print("last_expected_end is None, continue")
                        continue
                    elif last_expected_end is not None and actual_end - last_expected_end <= pd.Timedelta(seconds=30):
                        print(f"⏳ Свеча {last_5min_time} завершена. Идем дальше...")
                # --- 3. Проверяем, не обработали ли мы уже эту свечу ---
                if last_processed_time is not None and last_5min_time <= last_processed_time:
                    await asyncio.sleep(10)
                    continue

                # --- 4. Теперь можно действовать ---
                print(f"✅ Свеча {last_5min_time} завершена. Делаем предсказание...")
                obj = df_5min.index
                print("obj", obj)
                profit = self.backtest.quant_money(last_candle['close'], obj[-1])
                print("getCurrentProfit, ", profit)
                print("self.backtest.table", self.backtest.table)

                state = self._get_observation(df_5min)
                with torch.no_grad():
                    policy, value = worker_model(state)
                logits = policy.view(-1)
                action_dist = torch.distributions.Categorical(logits=logits)

                action = action_dist.sample().item()

                await self.step(Signal=action, prices=last_candle['close'])

                # --- 5. Запоминаем, что обработали эту свечу ---
                last_processed_time = last_5min_time

                # --- 6. Ждём следующей возможной свечи ---
                print("Ожидаем следующую 5-минутную свечу...")
                await asyncio.sleep(10)  # Можно увеличить, но не обязательно

            except Exception as e:
                print(f"❌ Ошибка: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

            #await asyncio.sleep(self.check_interval)
    async def send_daily_report(self):
        """Отправляет ежедневный отчёт в канал"""
        try:
            # Получаем текущие данные
            # Текущее время (без даты)
            now = datetime.now()
            date = now.date()  # datetime.time объект
            #print("self.backtest.table", self.backtest.table)
            #print("self.backtest.table Account_money", self.backtest.table['Account_money'])
            current_money = self.backtest.table['Account_money'][-1]
            print("current_money", current_money)
            start_money = self.settings.startCash
            total_profit = current_money - start_money
            print("total_profit", total_profit)
            daily_profit, percentage_change = self.get_daily_profit()  # Реализуем ниже
            #last_action_time = self.get_last_action_time()  # Можно хранить в Trader
            print("daily_profit, percentage_change", daily_profit, percentage_change)
            # Форматируем
            total_profit_str = f"+{total_profit:,.0f}" if total_profit >= 0 else f"{total_profit:,.0f}"
            daily_profit_str = f"+{daily_profit:,.0f}" if daily_profit >= 0 else f"{daily_profit:,.0f}"
            print("готовим отчет")
            report = (
                "📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b> | {date}\n\n"
                "💰 <b>Общая прибыль:</b> {total_profit} ₽\n"
                "🏦 <b>Текущий баланс:</b> {current_money:,.0f} ₽\n"
                "📈 <b>Прибыль за день:</b> {daily_profit} ₽\n"
                "📊 <b>Сделок сегодня:</b> {trades_today}\n"
                "🟢 <b>Покупок:</b> {buys}\n"
                "🔴 <b>Продаж:</b> {sells}\n"
                #"🕒 <b>Последняя сделка:</b> {last_time}\n\n"
                "✅ <i>Стратегия работает</i>"
            ).format(
                date=datetime.now().strftime("%d %B %Y"),
                total_profit=total_profit_str.replace(',', ' '),
                current_money=current_money,
                daily_profit=daily_profit_str.replace(',', ' '),
                trades_today=5,   # Пример — можно улучшить
                buys=3,
                sells=2,
                #last_time=last_action_time or "—"
            )

            await self.bot.send_message(
                chat_id=CHANNEL_ID,
                text=report,
                parse_mode="HTML"
            )
            print("✅ Ежедневный отчёт отправлен в канал")
        except Exception as e:
            print(f"❌ Ошибка при отправке отчёта: {e}")

    async def wait_for_daily_report(self):
        """Ждём 19:00 каждый день и отправляем отчёт"""
        print("❌❌❌❌❌❌❌❌❌self.is_running", self.is_running)
        day = None
        while self.is_running:
            now = datetime.now()
            print("⌚⌚⌚⌚⌚⌚⌚⌚time now", now)
            target_time = time(21, 2, 0)  # 19:00:00

            # Сегодняшняя дата + 19:00
            report_dt = datetime.combine(now.date(), target_time)

            # Если уже прошло 19:00 — ждём завтра
            if now.time() >= target_time:
                report_dt += timedelta(days=1)

            wait_seconds = (report_dt - now).total_seconds()
            print(f"⏳ Ожидание ежедневного отчёта до {report_dt.strftime('%Y-%m-%d %H:%M')}")

            # Ждём, но с проверкой stop_event
            """try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=wait_seconds)
                # Если пришёл сигнал остановки — выходим
                print(f"⏳⏳⏳ Ожидаем self.stop_event.wait()-------------------------------: {e}")
                return
            except Exception as e:
                print(f"❌ Ошибка при отправке отчёта: {e}")"""
                # Время пришло — отправляем отчёт
            if self.is_running and day != now.date() and now.time() >= target_time:
                 await self.send_daily_report()
                 day = now.date()
            else:
                await asyncio.sleep(600)
                continue

        print("Отчёт: цикл завершён")

    async def start(self):
        """Запуск стратегии начинается с этой функции."""
        
        if self.is_running:
            await self.bot.send_message(chat_id=self.chat_id, text="Стратегия уже запущена.")
            return

        self.is_running = True
        self.stop_event.clear()
        print("запуск стратегии async def start")
        action_numbers = 3
        BATCHSIZE = 1
        model = ActorCriticConv1D(2, 64, action_numbers, BATCHSIZE, 1)
        model.load_state_dict(torch.load(PATHFORMODELSPTH))  # загружаем обученную модель нейросети
        print("модель загружена")
        #await self.run_episode(worker_model=model)
        self.task = asyncio.create_task(self.run_episode(worker_model=model))  # запуск основного цикла стратегии
        self.report_task = asyncio.create_task(self.wait_for_daily_report())  # 🔥 Отчёт
        print("5. run_episode завершён (но он бесконечный, так что сюда не должно быть)")
        await self.bot.send_message(chat_id=self.chat_id, text="✅ Таски запущены.")

    async def stop(self):
        """Остановка стратегии."""
        if not self.is_running:
            await self.bot.send_message(chat_id=self.chat_id, text="Стратегия не запущена.")
            return

        self.is_running = False
        self.stop_event.set()  # Сигнал остановить цикл
        if self.task:
            await self.task
        if self.report_task:
            await self.report_task

        self.task = None
        self.report_task = None

        # Сохраним состояние при остановке
        self.backtest.save_state()

        # Отправим отчёт
        my_money = self.backtest.my_money
        posVolume = self.backtest.posVolume
        state = self.backtest.state
        profit = self.backtest.getCurrentProfit(0)  # Или передай последнюю цену
        mess = (
            "🛑 Стратегия остановлена.\n"
            f"Деньги: {my_money:.2f}\n"
            f"Позиция: {posVolume}\n"
            f"Состояние: {state}\n"
            f"Текущий профит: {profit:.2f}"
        )
        await self.bot.send_message(chat_id=self.chat_id, text=mess)
        print("Стратегия остановлена и состояние сохранено.")

async def run_strategy(bot: Bot, chat_id: int):
    """Запускаем асинхронно стратегию для каждого тикера из портфеля."""
    print("Запускаем асинхронно стратегию для каждого тикера из портфеля run_strategy(bot: Bot, chat_id: int)")
    async with aiohttp.ClientSession() as session:
        strategy_tasks = []
        # Загружаем модель один раз
        strategy = Strategy(settings=settings, session=session, bot=bot, chat_id=chat_id)
        command_stop_handler.strategy = strategy  # Сохраняем ссылку
        my_money = strategy.backtest.my_money
        posVolume = strategy.backtest.posVolume
        mess = (
                f"💰Свободные деньги: {round(my_money, 1)}.\n"
                f"⚖️Текущая позиция: {strategy.backtest.state}.\n"
                f"💳Цена при открытии позиции: {strategy.backtest.price_of_pos} ₽.\n"
                f"🛒Количество акций в портфеле ПАО \"Газпром\": {posVolume}."
            )
        
        await bot.send_message(chat_id="-1003065962569", text=mess)
        await bot.send_message(chat_id=chat_id, text=mess)
        # strategy_tasks.append(asyncio.create_task(strategy.start(model))) # Передаем модель
        # Запускаем каждую стратегию в отдельной задаче
        task = asyncio.create_task(strategy.start())
        print("task = asyncio.create_task(strategy.start()) сделано")
        strategy_tasks.append(task)
        await bot.send_message(chat_id=chat_id, text=f"Стратегия  запущена.")

        # Ждем завершения всех задач (теоретически никогда не завершится)
        await asyncio.gather(*strategy_tasks, return_exceptions=True)


# --- Обработчики команд ---

@router.message(Command(commands=["start"]))#сюда попадаем при обработке команды start
async def command_start_handler(message: Message) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    CHAT_ID = message.chat.id
    await message.answer(f"Привет, {message.from_user.full_name}! Бот запущен. ID чата: {CHAT_ID}")
    # Здесь можно запустить стратегию после получения chat_id
    # Но лучше это делать по отдельной команде, чтобы избежать повторного запуска

"""@router.message(Command(commands=["run"]))#сюда попадаем при обработке команды start
async def command_run_handler(message: Message) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    global CHAT_ID
    CHAT_ID = message.chat.id
    await message.answer(f"Привет, {message.from_user.full_name}! Бот запущен. ID чата: {CHAT_ID}")"""
    # Здесь можно запустить стратегию после получения chat_id
    # Но лучше это делать по отдельной команде, чтобы избежать повторного запуска

@router.message(Command(commands=["stop"]))
async def command_stop_handler(message: Message, bot: Bot) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    global CHAT_ID
    CHAT_ID = message.chat.id
    await message.answer("Останавливаю стратегию...")

    # Найдём запущенную стратегию
    # Упрощённо: предположим, что стратегия хранится в run_strategy
    # В реальности можно хранить стратегию в глобальной переменной или состоянии
    if hasattr(command_stop_handler, 'strategy') and command_stop_handler.strategy:
        await command_stop_handler.strategy.stop()
    else:
        await message.answer("Стратегия не найдена или не запущена.")

@router.message(Command(commands=["trade"]))
async def command_trade_handler(message: Message, bot: Bot) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    global CHAT_ID
    CHAT_ID = message.chat.id # Обновляем CHAT_ID на случай, если команда /start не была использована
    await message.answer("Запуск стратегии...")
    
    # Запускаем стратегию в фоновой задаче, чтобы не блокировать обработчик
    asyncio.create_task(run_strategy(bot, CHAT_ID))

# --- Основная точка входа ---
async def main() -> None:
    # Инициализация бота и диспетчера
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    # Создаем необходимые каталоги (если нужно)
    # functions.create_some_folders(timeframes=[Config.timeframe_0])
    print("Бот запущен. Ожидание команд...")
    # Запуск поллинга
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())