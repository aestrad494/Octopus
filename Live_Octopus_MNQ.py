#
# Python Script with live trading 
# Hermes Strategy Class
# 
# (c) Andres Estrada Cadavid
# QSociety

from Live_Class import Live
from Indicators import Indicators
import numpy as np
import pandas as pd
from datetime import datetime
from tzlocal import get_localzone
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam

class LiveOctopus(Live, Indicators):
    def trailing_stop(self, price_in, trailing, sl, order):
        '''Trailing Stop

        Parameters:
            price_in (float): entry price
            trailing (float): trailing stop level
            sl (float): current stop loss price level
            order (object): order stop loss
        
        Returns:
            float: updated stop loss price level
        '''
        new_sl = sl
        if trailing > 0:
            if self.position > 0:
                if self.x_round(self.data.iloc[-1].high) - price_in >= trailing:
                    if sl < self.x_round(self.data.iloc[-1].high) - trailing:
                        new_sl = round(self.x_round(self.data.iloc[-1].high) - trailing, self.digits)
                        order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract,order)
                        self.print('trailing stop from %5.2f to %5.2f'%(sl, new_sl))
            if self.position < 0:
                if price_in - self.x_round(self.data.iloc[-1].low) >= trailing:
                    if sl > self.x_round(self.data.iloc[-1].low) + trailing:
                        new_sl = round(self.x_round(self.data.iloc[-1].low) + trailing, self.digits)
                        order.auxPrice = new_sl
                        self.ib.placeOrder(self.contract,order)
                        self.print('trailing stop from %5.2f to %5.2f'%(sl, new_sl))
        return self.x_round(new_sl)
    
    def create_model(self, optimizer, hl=2, hu=128, dropout=False, rate=0.3, regularize=False, 
                    reg=l1(0.0005), input_dim=0):
        if not regularize:
            reg = None
        model = Sequential()
        model.add(Dense(hu, input_dim=input_dim, activity_regularizer=reg, activation='relu'))
        if dropout:
            model.add(Dropout(rate, seed=100))
        for _ in range(hl):
            model.add(Dense(hu, activation='relu', activity_regularizer=reg))
            if dropout:
                model.add(Dropout(rate, seed=100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def slope(self, data, columns, prev):
        for col in columns:
            slope = [0 for i in range(prev-1)]
            slope.extend([(data[col].iloc[i] - data[col].iloc[i-prev+1])/data[col].iloc[i-prev+1] for i in range(prev-1, len(data), 1)])
            data['%s_slope'%col] = slope
        return data

    def get_features(self, tempos, periods):
        features = []
        for tempo in tempos:
            feat = ['sub%s%s_%s'%(periods[0], periods[1], tempo), 'sub%s%s_%s'%(periods[0], periods[2], tempo), 
                    'sub%s%s_%s'%(periods[1], periods[2], tempo), '%s_slope_%s'%(periods[0], tempo),
                    '%s_slope_%s'%(periods[1], tempo), '%s_slope_%s'%(periods[2], tempo)]
            features.extend(feat)
        return features

    def add_features(self, data):
        # subtractions
        params = ['close', 'SMA_21', 'SMA_89']
        for i in range(len(params)):
            for j in range(len(params)):
                if i != j:
                    data['sub%s%s'%(params[i], params[j])] = data[params[i]] - data[params[j]]
        
        # slopes
        prev = 3
        data = self.slope(data, params, prev)
        '''## prices
        price_slope =  [0 for i in range(prev-1)]
        price_slope.extend([(data.close.iloc[i] - data.close.iloc[i-prev+1])/data.close.iloc[i-prev+1] for i in range(prev-1, len(data), 1)])
        data['price_slope'] = price_slope
        ## sma 21
        s21_slope =  [0 for i in range(prev-1)]
        s21_slope.extend([(data.SMA_21.iloc[i] - data.SMA_21.iloc[i-prev+1])/data.SMA_21.iloc[i-prev+1] for i in range(prev-1, len(data), 1)])
        data['s21_slope'] = s21_slope
        # sma 89
        s89_slope =  [0 for i in range(prev-1)]
        s89_slope.extend([(data.SMA_89.iloc[i] - data.SMA_89.iloc[i-prev+1])/data.SMA_89.iloc[i-prev+1] for i in range(prev-1, len(data), 1)])
        data['s89_slope'] = s89_slope'''

        return data
    
    def prints(self, message, type='a'):
        sample = open('brick_size.txt', type)
        print(message, file=sample)
        print(message)
        sample.close()

    def run_strategy(self, contracts, stop, target_1, target_2, trailing, periods, tempos, init, final):
        self.print('%s %s | Octopus 2 Contracts Bot Turned On' % (self.date, self.hour))
        self.print('%s %s | Running with stop: %d & target_1: %d & target_2: %d & trailing: %.2f'%
                      (self.date, self.hour, stop, target_1, target_2, trailing))
        # Check if operable schedule
        self.operable_schedule()

        if self.operable:
            # Defining Variables
            prediction = 0
            sent = False; exit_1 = False
            self.save_position()
            self.global_position = self.check_global_position()
            last_ib = 0
            #tempos = [60, 120]
            #periods = [21, 89]
            prev = 3
            max_tempos = max([int(t) for t in tempos])
            max_periods = max([int(periods[i][4:]) for i in range(1, len(periods))])
            #idx_back = int((max(periods)+5)*max(tempos)/5)
            idx_back = int((max_periods+5)*max_tempos/5)
            ana_time = int(max_tempos / 60)

            features = self.get_features(tempos, periods)
            '''features = ['subcloseSMA_21_60', 'subcloseSMA_89_60', 'subSMA_21SMA_89_60',
                        'price_slope_60', 's21_slope_60', 's89_slope_60', 
                        'subcloseSMA_21_120', 'subcloseSMA_89_120', 'subSMA_21SMA_89_120',
                        'price_slope_120', 's21_slope_120', 's89_slope_120']'''
            n_features = len(features)
            features_1 = features[:int(n_features/2)]
            features_2 = features[int(n_features/2):]
            lags = 1

            parameters = pd.read_csv('parameters/parameters_%s.csv'%self.symbol)
            layers = parameters.layers[0]
            hidden_units = parameters.hidden_units[0]
            learning_rate = parameters.learning_rate[0]
            dropout = parameters.dropout[0]

            optimizer = Adam(learning_rate=learning_rate)
            model = self.create_model(optimizer=optimizer, hl=layers, hu=hidden_units, dropout=dropout, input_dim=n_features)
            #model.load_weights('model/octopus_model_%s.h5f'%self.symbol)
            model.load_weights('model/octopus_model_%s_%s_%s_%s_%s.h5f'%(self.symbol, init, final, tempos[0], tempos[1]))
            print('model loaded!')

            #target condition
            if target_1 == target_2: target_2 += 1

            # Getting Historical Data to get renko bars
            self.print('Downloading Historical %s Data...'%self.symbol)
            self.data = self.get_historical_data()
            print(self.data)
            self.print('Historical Data retrieved! from %s to %s'%(self.data.index[0], self.data.index[-1]))

            #############################################################################

            '''data_60 = self.resampler(self.data.iloc[-idx_back:], '60S', type='bars')
            data_120 = self.resampler(self.data.iloc[-idx_back:], '120S', type='bars')

            for per in periods:
                data_60['SMA_%d'%per] = self.SMA(data_60.close, per)
                data_120['SMA_%d'%per] = self.SMA(data_120.close, per)
            
            data_60_eval = self.add_features(data_60)
            data_120_eval = self.add_features(data_120)

            data_60_eval.columns = ['%s_60'%col for col in data_60_eval.columns]
            data_120_eval.columns = ['%s_120'%col for col in data_120_eval.columns]

            model_input = list(data_60_eval.iloc[-1][features_60].values)
            model_input.extend(list(data_120_eval.iloc[-1][features_120].values))
            model_input = np.reshape(model_input, [1, lags, n_features])

            prediction = 1 if model.predict(model_input)[0][0][0] > 0.5 else 0
            
            print(prediction)'''

            #############################################################################
            
            for kafka_bar in self.consumer:
                # Concatening last bar kafka method
                try:
                    kafka_bar_df = pd.DataFrame(eval(str(kafka_bar.value, encoding='utf-8')), index=[0]).set_index('time')
                    kafka_bar_df.index = pd.to_datetime(kafka_bar_df.index)
                    self.data = pd.concat([self.data, kafka_bar_df])
                except: pass

                self.current_date()
                self.ib.sleep(1)
                self.continuous_check_message('%s %s | %s %s is running OK. Last price: %.2f' % 
                                                    (self.date, self.hour, self.bot_name, self.symbol, self.x_round(self.data.iloc[-1].close)))
                self.daily_results_positions()         # Send daily profit message to telegram
                self.weekly_metrics()                  # Send week metrics message to telegram

                # Check Global Position
                try: self.global_position = self.check_global_position()
                except: self.ib.sleep(2); self.global_position = self.check_global_position()

                # Check if it's time to reconnect
                self.connected = self.ib.isConnected()
                self.reconnection()

                if self.connected:
                    # Check for Entry
                    if self.position == 0 and self.global_position == 0:                # if there's not opened positions
                        if pd.to_datetime(self.hour).minute % ana_time:
                            prediction = 0
                            data_1 = self.resampler(self.data.iloc[-idx_back:], tempos[0]+'S', type='bars')
                            data_2 = self.resampler(self.data.iloc[-idx_back:], tempos[1]+'S', type='bars')

                            pers = periods[1:]
                            for per in pers:
                                per = int(per[4:])
                                data_1['SMA_%d'%per] = self.SMA(data_1.close, per)
                                data_2['SMA_%d'%per] = self.SMA(data_2.close, per)
                            
                            data_1_eval = self.add_features(data_1)
                            data_2_eval = self.add_features(data_2)

                            data_1_eval.columns = ['%s_%s'%(col,tempos[0]) for col in data_1_eval.columns]
                            data_2_eval.columns = ['%s_%s'%(col,tempos[1]) for col in data_2_eval.columns]

                            model_input = list(data_1_eval.iloc[-1][features_1].values)
                            model_input.extend(list(data_2_eval.iloc[-1][features_2].values))
                            model_input = np.reshape(model_input, [1, lags, n_features])
                            

                            prediction = 1 if model.predict(model_input)[0][0][0] > 0.5 else 0
                            self.print('%s : %d'%(model_input, prediction))
                            
                        # Entry conditions
                        if not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('16:00:00').time()):
                            ## Buys
                            if not sent and prediction > 0:
                                max_stop = stop*self.leverage*contracts
                                price_buy_in_1, sl_buy_1, tp_buy_1, time_buy_in, comm_buy_in_1, profit_buy, ord_buy_sl_1, ord_buy_tp_1 = self.braket_market('BUY', contracts/2, stop, target_1, max_stop)
                                price_buy_in_2, sl_buy_2, tp_buy_2, time_buy_in, comm_buy_in_2, profit_buy, ord_buy_sl_2, ord_buy_tp_2 = self.braket_market('BUY', contracts/2, stop, target_2, max_stop, entry_price=price_buy_in_1)     
                                if price_buy_in_1 > 0 and price_buy_in_2 > 0: sent = True
                                bar_entry = len(movements)
                                tr_1 = self.x_round(trailing * target_1)
                                tr_2 = self.x_round(trailing * target_2)
                                exit_1 = False

                    # Check for Exit
                    if self.position > 0:
                        try:
                            if not exit_1: sl_buy_1 = self.trailing_stop(price_in=price_buy_in_1, trailing=tr_1, sl=sl_buy_1, order=ord_buy_sl_1)
                            sl_buy_2 = self.trailing_stop(price_in=price_buy_in_1, trailing=tr_2, sl=sl_buy_2, order=ord_buy_sl_2)
                        except: self.print('Trying to apply trailing stop... Order has been Filled!')

                        # By stop ==========
                        ## Stop 1
                        if self.check_pendings(ord_buy_sl_1) and not exit_1 and self.position > 0:   # Check if stop 1 is filled
                            self.exit_pending(ord_buy_sl_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'sl1')
                            exit_1 = True; sent = False
                        ## Stop 2
                        if self.check_pendings(ord_buy_sl_2) and self.position > 0:   # Check if stop 2 is filled
                            self.exit_pending(ord_buy_sl_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'sl2')
                            sent = False
                        
                        ## False Stop
                        ### Stop 1
                        if not self.check_pendings(ord_buy_sl_1) and sl_buy_1 - self.data.iloc[-1].low >= 2 and not exit_1 and self.position > 0:
                            self.exit_market(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'fsl1')
                            exit_1 = True; sent = False
                        ### Stop 2
                        if not self.check_pendings(ord_buy_sl_2) and sl_buy_2 - self.data.iloc[-1].low >= 2 and self.position > 0:
                            self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fsl2')
                            sent = False
                        
                        # By target ==========
                        ## Target 1
                        if self.check_pendings(ord_buy_tp_1) and not exit_1 and self.position > 0:    # Check if target 1 is filled
                            self.exit_pending(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'tp1')
                            exit_1 = True; sent = False
                        ## Target 2
                        if self.check_pendings(ord_buy_tp_2) and self.position > 0:                   # Check if target 2 is filled
                            self.exit_pending(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'tp2')
                            sent = False

                        # Exit on friday
                        if self.weekday == 4 and pd.to_datetime(self.hour).time() >= pd.to_datetime('16:57:00').time() and self.position > 0:
                            if self.position == contracts:
                                self.exit_market(ord_buy_tp_1, 'BUY', contracts/2, price_buy_in_1, time_buy_in, comm_buy_in_1, 'fri')
                                self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fri')
                                sent = False
                            else:
                                self.exit_market(ord_buy_tp_2, 'BUY', contracts/2, price_buy_in_2, time_buy_in, comm_buy_in_2, 'fri')
                                sent = False

                else:
                    if not self.interrumption:
                        try:
                            self.print('Trying to reconnect...')
                            self.ib.disconnect()
                            self.ib.sleep(10)
                            self.ib.connect('127.0.0.1', self.port, self.client)
                            self.connected = self.ib.isConnected()
                            if self.connected:
                                self.print('Connection reestablished!')
                                self.print('Getting Data...')
                                self.data = self.get_historical_data()
                                renko_object.build_history(prices=self.data.close.values, dates=self.data.index)
                                prices = renko_object.get_renko_prices()
                        except:
                            self.print('Connection Failed! Trying to reconnect in 10 seconds...')

            self.ib.disconnect()
            self.print('%s %s | Session Ended. Good Bye!' % (self.date, self.hour))

if __name__ == '__main__':
    symbol = 'NQ'
    port = 7497
    client = 152

    live_octopus = LiveOctopus(symbol=symbol, bot_name='Octopus (demo)', temp='1 min', port=port, client=client, real=False)
    
    init = '2021-10-16'
    final = '2021-11-20'
    periods = ['close', 'SMA_21', 'SMA_89']
    tempos = ['60', '120']          #['180', '240'] ['540', '720']
    live_octopus.run_strategy(contracts=2, stop=3, target_1=3, target_2=3, trailing=0.7,
                                periods=periods, tempos=tempos, init=init, final=final)
