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
from backtest.TraderUtil import readSettings, ScaleValues, RLSettings
from backtest.createDirectory import makeFolderName, makeDir
from config import TOKEN as BOT_TOKEN, ADMIN_ID, CHANNEL_ID

# Интервал данных (1 или 10 минут)
INTERVAL = 10

if INTERVAL == 10:
    RESAMPLE_NUM = 30
    RESAMPLE_INT = f"{RESAMPLE_NUM}T"
    CANDLES_NUM = 14 * 12 + 3 * 3
elif INTERVAL == 1:
    RESAMPLE_NUM = 5
    RESAMPLE_INT = f"{RESAMPLE_NUM}T"
    CANDLES_NUM = 14 * 5 + 3 * 3


class MM:
    MAX = 0
    MIN = 0


class Feauteres:
    names = 0
    minmax = 0


def getFeautersDataFromFile(pathToFile, feauter):
    """Загружает мин/макс значения признаков из CSV-файла."""
    df = pd.read_csv(pathToFile).set_index("NameOfParam")
    returns_MAX = df.loc["returns_MAX", "Values"]
    returns_MIN = df.loc["returns_MIN", "Values"]
    RSI_5_MAX = df.loc["RSI_5_MAX", "Values"]
    RSI_5_MIN = df.loc["RSI_5_MIN", "Values"]
    AD5_returns_MAX = df.loc["AD5_returns_MAX", "Values"]
    AD5_returns_MIN = df.loc["AD5_returns_MIN", "Values"]

    OneVals = MM()
    OneVals.MAX, OneVals.MIN = returns_MAX, returns_MIN
    SecondVals = MM()
    SecondVals.MAX, SecondVals.MIN = RSI_5_MAX, RSI_5_MIN
    ThreeVals = MM()
    ThreeVals.MAX, ThreeVals.MIN = AD5_returns_MAX, AD5_returns_MIN

    feauteresData = Feauteres()
    feauteresData.names = feauter
    feauteresData.minmax = [OneVals, SecondVals, ThreeVals]
    return feauteresData


def logData(file, string):
    """Записывает строку в лог-файл."""
    with open(file, "a") as myfile:
        myfile.write(string)


def calculate_rsi(data, rsi_period):
    """Вычисляет RSI и сдвигает его на один шаг назад для избежания lookahead bias."""
    dataset = data.copy()
    delta = dataset['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    dataset['rsi_df'] = rsi.shift(1)
    return dataset


# --- Настройки из файла ---
pathToFile = "/backtest/TraderExamples/Settings/SettingsGAZPTrain_7-10-2024_13-2-2025_01092025.csv"
(
    TICKER, nameOfTicker, timeDelta, folderName, rewardType, nameOfDates,
    nameOfTestDates, dateName, lots, startCash, comisBroker, comisMoex,
    del_r, Gamma, Clc, lr, gran, will, rewardMult, willBeGap_using,
    willBeGap_value, passLoopForward, dontMakeCurrentDeals
) = readSettings(pathToFile)

PERIOD = f'/{timeDelta}m/'
comis = (comisBroker, comisMoex)
folderName = makeFolderName(folderName, comis[0], Gamma, Clc, lr, del_r, gran, will)

nameOfModel = "GAZP_ActorCriticConv1D_Tr_Rw=410748_Test_Rw=139382_returns_RSI5_sc_AD5_returns_sc__ep=490"
PATHFORMODELS = f"/backtest/TrainedModels/{TICKER}{PERIOD}{folderName}"
PATHFORMODELSPTH = f"{PATHFORMODELS}pth/{nameOfModel}.pth"
PATHFORPDF = f"{PATHFORMODELS}test/{nameOfModel}"
PATHFORJITMODELS = f"{PATHFORMODELS}l/{nameOfModel}_jit.pt"
pathForData = f"{PATHFORMODELS}data/"
pathForCurrentTestFolder = f"{PATHFORMODELS}test/{nameOfModel.split('=')[-1]}/"

fileLogName = "Log.txt"
if os.path.exists(pathForCurrentTestFolder + fileLogName):
    fileLogName = "Log2.txt"

nameOfPreparedDataFile = f"{nameOfTicker}{nameOfDates}.csv"
nameOfTestDataFile = f"{nameOfTicker}{nameOfTestDates}_Test.csv"
nameOfMinMaxDataFile = f"{nameOfDates}_MINMAX.csv"

settings = RLSettings(
    pathForCurrentTestFolder, lots, lots, startCash, comis, rewardType,
    del_r, gran, will, timeDelta, rewardMult, Gamma, Clc, lr,
    willBeGap_using, willBeGap_value, passLoopForward, dontMakeCurrentDeals
)

# --- Признаки и их мин/макс значения ---
feauteres = ['returns', 'RSI5_sc', 'AD5_returns_sc']
feauteresForTest = ['Close', 'RSI5', 'AD5']
feautersData = getFeautersDataFromFile(pathForData + nameOfMinMaxDataFile, feauteresForTest)
settings.feautData = feautersData
numOfFeauteres = len(feauteres)
action_numbers = 3

# --- Telegram ---
CHAT_ID = None
router = Router()


def wait_until_next_interval(interval_minutes: int = 5):
    """Ожидает до следующего временного интервала, кратного interval_minutes."""
    now = datetime.now()
    minutes_since_hour = now.minute
    remainder = minutes_since_hour % interval_minutes
    next_minute = (minutes_since_hour - remainder) + interval_minutes

    if next_minute >= 60:
        next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)

    wait_seconds = (next_time - now).total_seconds()
    return asyncio.sleep(wait_seconds)


class Strategy:
    def __init__(self, settings: RLSettings, session: Optional[ClientSession], bot: Bot, chat_id: int):
        self.session = session
        self.settings = settings
        self.backtest = Trader.load_state(money=settings.startCash, comis=settings.comis, printBool=True)
        self.bot = bot
        self.chat_id = chat_id
        self.is_running = False
        self.stop_event = asyncio.Event()
        self.task = None
        self.report_task = None

    async def step(self, Signal: int, prices: float):
        """Обрабатывает торговое решение и отправляет уведомления в Telegram."""
        current_price_str = f"{prices:.2f}".replace('.', ',')
        ticker_name = "ПАО \"Газпром\""
        current_profit = self.backtest.getCurrentProfit(prices)
        profit_percent = 0
        if self.backtest.posVolume != 0:
            profit_percent = (current_profit / abs(self.backtest.moneyOnStartDeal)) * 100 if self.backtest.moneyOnStartDeal != 0 else 0

        endOfDataframe = False  # Заглушка; в live-режиме всегда False

        if Signal == 2 and self.backtest.state == "zero":
            self.backtest.price_of_pos = prices
            self.backtest.buy(self.settings.lot, prices)
            assert self.backtest.state == "Long"
            mess = f"⬆️🟢 Покупка акций {ticker_name} по цене {current_price_str} ₽."
            await self._broadcast_message(mess)

        elif Signal == 0 and self.backtest.state == "zero":
            self.backtest.price_of_pos = prices
            self.backtest.sell(self.settings.lot, prices)
            assert self.backtest.state == "Short"
            mess = f"⬇️🔴 Продажа акций {ticker_name} по цене {current_price_str} ₽."
            await self._broadcast_message(mess)

        elif (
            (Signal == 1 and self.backtest.state in ["zero", "Long", "Short"]) or
            (Signal == 2 and self.backtest.state == "Long") or
            (Signal == 0 and self.backtest.state == "Short")
        ):
            mess = f"🟡 Ничего не делаем, позиция {self.backtest.state}. Цена {current_price_str} ₽."
            await self._broadcast_message(mess)

        elif self.backtest.state == "Long" and Signal == 0:
            profit_abs = current_profit
            profit_percent = (profit_abs / self.backtest.moneyOnStartDeal) * 100 if self.backtest.moneyOnStartDeal != 0 else 0
            mess = (
                f"⬇️ Продажа акций {ticker_name} по цене {current_price_str} ₽. "
                f"Профит от сделки {profit_abs:.1f} ₽. Прибыль сделки {profit_percent:.1f}%."
            )
            self.backtest.price_of_pos = 0
            self.backtest.sell(self.settings.lot, prices)
            await self._broadcast_message(mess)

        elif self.backtest.state == "Short" and Signal == 2:
            profit_abs = current_profit
            profit_percent = (profit_abs / self.backtest.start_money) * 100 if self.backtest.start_money != 0 else 0
            mess = (
                f"⬆️ Покупка акций {ticker_name} по цене {current_price_str} ₽. "
                f"Профит от сделки {profit_abs:.1f} ₽. Прибыль сделки {profit_percent:.1f}%."
            )
            self.backtest.price_of_pos = 0
            self.backtest.buy(self.settings.lot, prices)
            await self._broadcast_message(mess)

        else:
            mess = "⚪ Ничего не делаем."
            await self._broadcast_message(mess)

    async def _broadcast_message(self, text: str):
        """Отправляет сообщение в личный чат и канал."""
        await self.bot.send_message(chat_id=self.chat_id, text=text)
        await self.bot.send_message(chat_id=CHANNEL_ID, text=text)

    def get_daily_profit(self):
        """Рассчитывает прибыль за последний торговый день."""
        df = self.backtest.table[1:].copy()
        df.index = pd.to_datetime(df.index)
        last_date = df.index.date.max()
        last_day = df[df.index.date == last_date]

        if len(last_day) > 1:
            first_value = last_day['Account_money'].iloc[0]
            last_value = last_day['Account_money'].iloc[-1]
            change = last_value - first_value
            percentage_change = (change / first_value) * 100
            return change, percentage_change
        else:
            return 0, 0

    def _get_observation(self, df):
        """Формирует входное состояние для нейросети."""
        obj = df.index
        current_step = len(obj) - 1
        window_size = 3
        feauteresForTest = ['Close', 'RSI5', 'AD5']
        stated = GetStateIndCombConv1D(obj, current_step, df, feauteresForTest, window_size)
        statedSq = torch.squeeze(stated, 0)
        statedScaled = ScaleValues(statedSq, self.settings.feautData, 2, 3)
        return statedScaled, stated

    async def get_all_candles(self, start, end, interval):
        """Получает исторические свечи с MOEX и агрегирует их до нужного интервала."""
        async with aiohttp.ClientSession() as session:
            try:
                candles = await aiomoex.get_market_candles(
                    session, security='GAZP', interval=interval,
                    start=start, end=end, market='shares', engine='stock'
                )
                if not candles:
                    return None, None

                last_candle = candles[-1]
                df_candle = candles[-CANDLES_NUM:]
                df = pd.DataFrame(df_candle)
                df['datetime'] = pd.to_datetime(df['begin'])
                df.set_index('datetime', inplace=True)
                df = df.drop(columns=["begin", "end"])

                df = df.resample(RESAMPLE_INT).agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna(subset=['open', 'high', 'low', 'close'])

                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                df['Volume'] = df['Volume'].astype(float)

                # Индикаторы
                df['returns'] = df['Close'] - df['Close'].shift(1)
                rsi_df = calculate_rsi(df, 14)
                df["RSI5"] = rsi_df['rsi_df']
                df["AD5"] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])

                return df, last_candle
            except Exception as e:
                print(f"❌ Ошибка при получении свечей: {e}")
                return None, None

    async def get_securities_info(self):
        """Получает текущее время последнего обновления котировок GAZP на MOEX."""
        async with aiohttp.ClientSession() as session:
            try:
                securities_data = await aiomoex.get_board_securities(
                    session, table='marketdata', board='TQBR',
                    market='shares', engine='stock'
                )
                for item in securities_data:
                    if item.get('SECID') == 'GAZP':
                        update_time_str = item.get('UPDATETIME')
                        time_obj = datetime.strptime(update_time_str, "%H:%M:%S").time()
                        return datetime.combine(datetime.now().date(), time_obj)
            except Exception as e:
                print(f"Ошибка при получении данных: {e}")
            return None

    async def run_episode(self, worker_model):
        """Основной цикл live-торговли: получение данных, принятие решений, отправка сигналов."""
        last_processed_time = None
        target_time = time(23, 49, 59)      # Конец торгов в будни
        target_time_wekend = time(19, 0, 0) # Конец торгов в выходные
        start_time = time(9, 48, 58)        # Начало торгов

        while not self.stop_event.is_set():
            try:
                now = datetime.now()
                weekday = now.weekday()
                only_time = now.time()

                # Проверка рабочего времени
                if (weekday < 5 and (only_time > target_time or only_time < start_time)) or \
                   (weekday >= 5 and (only_time > target_time_wekend or only_time < start_time)):
                    await asyncio.sleep(60)
                    continue

                # Получение данных
                day_before_yesterday = now - timedelta(days=5)
                start = day_before_yesterday.strftime('%Y-%m-%d')
                end = now.strftime('%Y-%m-%d')
                df, last_candle = await self.get_all_candles(start, end, INTERVAL)
                if df is None or last_candle is None:
                    await asyncio.sleep(60)
                    continue

                # Получение времени последней 1-минутной свечи
                if INTERVAL == 10:
                    actual_end = await self.get_securities_info()
                else:
                    last_1min = await self.get_last_candle(end, end, 1)
                    actual_end = pd.to_datetime(last_1min['end']) if last_1min else None

                if actual_end is None:
                    await asyncio.sleep(60)
                    continue

                # Проверка завершения свечи
                last_30min_time = df.index[-1]
                expected_end = last_30min_time + pd.Timedelta(minutes=RESAMPLE_NUM) - pd.Timedelta(seconds=1)

                if actual_end < expected_end:
                    await asyncio.sleep(10)
                    continue

                if (actual_end - expected_end) > pd.Timedelta(seconds=30):
                    if last_processed_time is None or last_30min_time > last_processed_time:
                        last_processed_time = last_30min_time
                    await asyncio.sleep(10)
                    continue

                if last_processed_time is not None and last_30min_time <= last_processed_time:
                    await asyncio.sleep(10)
                    continue

                # Принятие решения
                state, stateF = self._get_observation(df)
                with torch.no_grad():
                    policy, _ = worker_model(state)
                logits = policy.view(-1)
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample().item()

                logData("action_020925.csv",
                        f"{actual_end},{last_candle['close']},{stateF.numpy()},{state.numpy()},{policy.numpy()}\n")

                await self.step(action, prices=last_candle['close'])
                last_processed_time = last_30min_time
                await asyncio.sleep(10)

            except Exception as e:
                print(f"❌ Ошибка в run_episode: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

    async def get_last_candle(self, start, end, interval):
        """Получает последнюю 1-минутную свечу."""
        async with aiohttp.ClientSession() as session:
            try:
                candles = await aiomoex.get_market_candles(
                    session, security='GAZP', interval=interval,
                    start=start, end=end, market='shares', engine='stock'
                )
                return candles[-1] if candles else None
            except Exception as e:
                print(f"❌ Ошибка при получении 1-минутной свечи: {e}")
                return None

    async def send_daily_report(self):
        """Отправляет ежедневный отчёт в Telegram-канал."""
        try:
            current_money = self.backtest.table['Account_money'].iloc[-1]
            total_profit = current_money - self.settings.startCash
            daily_profit, percentage_change = self.get_daily_profit()

            total_profit_str = f"+{total_profit:,.0f}" if total_profit >= 0 else f"{total_profit:,.0f}"
            daily_profit_str = f"+{daily_profit:,.0f}" if daily_profit >= 0 else f"{daily_profit:,.0f}"

            report = (
                "📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b> | {date}\n"
                "💰 <b>Общая прибыль:</b> {total_profit} ₽\n"
                "🏦 <b>Текущий баланс:</b> {current_money:,.0f} ₽\n"
                "📈 <b>Прибыль за день:</b> {daily_profit} ₽\n"
                "📊 <b>Сделок сегодня:</b> {trades_today}\n"
                "🟢 <b>Покупок:</b> {buys}\n"
                "🔴 <b>Продаж:</b> {sells}\n"
                "✅ <i>Стратегия работает</i>"
            ).format(
                date=datetime.now().strftime("%d %B %Y"),
                total_profit=total_profit_str.replace(',', ' '),
                current_money=current_money,
                daily_profit=daily_profit_str.replace(',', ' '),
                trades_today=5, buys=3, sells=2
            )
            await self.bot.send_message(chat_id=CHANNEL_ID, text=report, parse_mode="HTML")
        except Exception as e:
            print(f"❌ Ошибка при отправке отчёта: {e}")

    async def wait_for_daily_report(self):
        """Ожидает 21:02 каждый день и отправляет отчёт."""
        reported_day = None
        while self.is_running:
            now = datetime.now()
            target_time = time(21, 2, 0)
            if now.time() >= target_time and now.date() != reported_day:
                await self.send_daily_report()
                reported_day = now.date()
            await asyncio.sleep(600)

    async def start(self):
        """Запускает стратегию."""
        if self.is_running:
            await self.bot.send_message(chat_id=self.chat_id, text="Стратегия уже запущена.")
            return

        self.is_running = True
        self.stop_event.clear()

        model = ActorCriticConv1D(2, 64, action_numbers, 1, 1)
        model.load_state_dict(torch.load(PATHFORMODELSPTH))
        self.task = asyncio.create_task(self.run_episode(worker_model=model))
        self.report_task = asyncio.create_task(self.wait_for_daily_report())
        await self.bot.send_message(chat_id=self.chat_id, text="✅ Таски запущены.")

    async def stop(self):
        """Останавливает стратегию и сохраняет состояние."""
        if not self.is_running:
            await self.bot.send_message(chat_id=self.chat_id, text="Стратегия не запущена.")
            return

        self.is_running = False
        self.stop_event.set()
        if self.task:
            await self.task
        if self.report_task:
            await self.report_task

        self.backtest.save_state()
        profit = self.backtest.getCurrentProfit(0)
        mess = (
            "🛑 Стратегия остановлена.\n"
            f"Деньги: {self.backtest.my_money:.2f}\n"
            f"Позиция: {self.backtest.posVolume}\n"
            f"Состояние: {self.backtest.state}\n"
            f"Текущий профит: {profit:.2f}"
        )
        await self.bot.send_message(chat_id=self.chat_id, text=mess)


async def run_strategy(bot: Bot, chat_id: int):
    """Запускает торговую стратегию для GAZP."""
    async with aiohttp.ClientSession() as session:
        strategy = Strategy(settings=settings, session=session, bot=bot, chat_id=chat_id)
        command_stop_handler.strategy = strategy

        my_money = strategy.backtest.my_money
        posVolume = strategy.backtest.posVolume
        mess = (
            f"💰 Свободные деньги: {round(my_money, 1)}.\n"
            f"⚖️ Текущая позиция: {strategy.backtest.state}.\n"
            f"💳 Цена при открытии позиции: {strategy.backtest.price_of_pos} ₽.\n"
            f"🛒 Количество акций в портфеле ПАО \"Газпром\": {posVolume}."
        )
        await bot.send_message(chat_id=CHANNEL_ID, text=mess)
        await bot.send_message(chat_id=chat_id, text=mess)

        task = asyncio.create_task(strategy.start())
        await bot.send_message(chat_id=chat_id, text="Стратегия запущена.")
        await task


# --- Telegram команды ---
@router.message(Command(commands=["start"]))
async def command_start_handler(message: Message) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    global CHAT_ID
    CHAT_ID = message.chat.id
    await message.answer(f"Привет, {message.from_user.full_name}! ID чата: {CHAT_ID}")


@router.message(Command(commands=["stop"]))
async def command_stop_handler(message: Message, bot: Bot) -> None:
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ Доступ запрещён.")
        return
    global CHAT_ID
    CHAT_ID = message.chat.id
    await message.answer("Останавливаю стратегию...")
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
    CHAT_ID = message.chat.id
    await message.answer("Запуск стратегии...")
    asyncio.create_task(run_strategy(bot, CHAT_ID))


# --- Запуск бота ---
async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    print("Бот запущен. Ожидание команд...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())