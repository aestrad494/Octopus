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

    def run_strategy(self, contracts, stop_1, target_1, target_2, trailing_1, stop_2, target_3, target_4, trailing_2, stop_3, target_5, target_6, trailing_3, periods, tempos, init, final):
        self.print('%s %s | Octopus %d Contracts Bot Turned On' % (self.date, self.hour, contracts))
        self.print('%s %s | Running with stop_1: %d & target_1: %d & target_2: %d & trailing_1: %.2f & \
            stop_2: %d & target_3: %d & target_4: %d & trailing_2: %.2f & \
            stop_3: %d & target_5: %d & target_6: %d & trailing_3: %.2f '%
                      (self.date, self.hour, stop_1, target_1, target_2, trailing_1,
                                            stop_2, target_3, target_4, trailing_2,
                                            stop_3, target_5, target_6, trailing_3))
        # Check if operable schedule
        self.operable_schedule()

        if self.operable:
            # Defining Variables
            prediction_1 = 0; prediction_2 = 0; prediction_3 = 0
            sent = False; first = False; second = False; third = False; fourth = False; fifth = False; sixth = False
            self.save_position()
            self.global_position = self.check_global_position()
            prev = 3
            #max_tempos = max([int(t) for t in tempos])
            #max_tempos = max([int(tem[1]) for tem in tempos])
            max_tempos_1 = int(tempos[0][1])
            max_tempos_2 = int(tempos[1][1])
            max_tempos_3 = int(tempos[2][1])

            max_periods = max([int(periods[i][4:]) for i in range(1, len(periods))])
            #idx_back = int((max(periods)+5)*max(tempos)/5)
            #idx_back = int((max_periods+5)*max_tempos/5)
            idx_back_1 = int((max_periods+5)*max_tempos_1/5)
            idx_back_2 = int((max_periods+5)*max_tempos_2/5)
            idx_back_3 = int((max_periods+5)*max_tempos_3/5)
            # ana_time = int(max_tempos / 60)

            feat_1 = self.get_features(tempos[0], periods)
            feat_2 = self.get_features(tempos[1], periods)
            feat_3 = self.get_features(tempos[2], periods)
            # features = ['subcloseSMA_21_60', 'subcloseSMA_89_60', 'subSMA_21SMA_89_60',
            #             'price_slope_60', 's21_slope_60', 's89_slope_60', 
            #             'subcloseSMA_21_120', 'subcloseSMA_89_120', 'subSMA_21SMA_89_120',
            #             'price_slope_120', 's21_slope_120', 's89_slope_120']
            n_features = len(feat_1)
            features_1 = feat_1[:int(n_features/2)]
            features_2 = feat_1[int(n_features/2):]
            features_3 = feat_2[:int(n_features/2)]
            features_4 = feat_2[int(n_features/2):]
            features_5 = feat_3[:int(n_features/2)]
            features_6 = feat_3[int(n_features/2):]
            lags = 1
            allow_entry = True

            models = []
            for i in range(3):
                temp = tempos[i]
                parameters = pd.read_csv('parameters/parameters_%s_%s_%s.csv'%(self.symbol, temp[0], temp[1]))

                layers = parameters.layers[0]
                hidden_units = parameters.hidden_units[0]
                learning_rate = parameters.learning_rate[0]
                dropout = parameters.dropout[0]

                ini = init[i]
                optimizer = Adam(learning_rate=learning_rate)
                model = self.create_model(optimizer=optimizer, hl=layers, hu=hidden_units, dropout=dropout, input_dim=n_features)
                print('model/octopus_model_%s_%s_%s_%s_%s.h5f'%(self.symbol, ini, final, temp[0], temp[1]))
                model.load_weights('model/octopus_model_%s_%s_%s_%s_%s.h5f'%(self.symbol, ini, final, temp[0], temp[1]))
                models.append(model)
            
            model_1 = models[0]
            model_2 = models[1]
            model_3 = models[2]
            print('models loaded!')


            '''parameters = pd.read_csv('parameters/parameters_%s_%s_%s.csv'%(self.symbol, tempos[0], tempos[1]))
            layers = parameters.layers[0]
            hidden_units = parameters.hidden_units[0]
            learning_rate = parameters.learning_rate[0]
            dropout = parameters.dropout[0]'''

            '''optimizer = Adam(learning_rate=learning_rate)
            model = self.create_model(optimizer=optimizer, hl=layers, hu=hidden_units, dropout=dropout, input_dim=n_features)
            #model.load_weights('model/octopus_model_%s.h5f'%self.symbol)
            model.load_weights('model/octopus_model_%s_%s_%s_%s_%s.h5f'%(self.symbol, init, final, tempos[0], tempos[1]))
            print('model loaded!')'''

            #target condition
            if target_1 > target_2: target_2 = target_1
            if target_1 == target_2: target_2 += 1

            if target_3 > target_4: target_4 = target_3
            if target_3 == target_4: target_4 += 1

            if target_5 > target_6: target_6 = target_5
            if target_5 == target_6: target_6 += 1

            # Getting Historical Data to get renko bars
            self.print('Downloading Historical %s Data...'%self.symbol)
            self.data = self.get_historical_data()
            print(self.data)
            self.print('Historical Data retrieved! from %s to %s'%(self.data.index[0], self.data.index[-1]))

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
                        hour = pd.to_datetime(self.hour).hour
                        minute = pd.to_datetime(self.hour).minute
                        second = pd.to_datetime(self.hour).second
                        # if pd.to_datetime(self.hour).minute % ana_time == 0 and (second == 0 or second == 5):
                        if ((hour in [0, 3, 6, 9, 12, 15, 18, 21] and minute in [36, 0]) or \
                           (hour in [1, 4, 7, 10, 13, 16, 19, 22] and minute in [12, 48]) or \
                           (hour in [2, 5, 8, 11, 14, 17, 20, 23] and minute in [24])) and (second == 0 or second == 5):
                        # if minute % 2 == 0 and (second == 0 or second == 5):
                            prediction_1 = 0; prediction_2 = 0; prediction_3 = 0
                            data_1 = self.resampler(self.data.iloc[-idx_back_1:], tempos[0][0]+'S', type='bars')
                            data_2 = self.resampler(self.data.iloc[-idx_back_1:], tempos[0][1]+'S', type='bars')
                            data_3 = self.resampler(self.data.iloc[-idx_back_2:], tempos[1][0]+'S', type='bars')
                            data_4 = self.resampler(self.data.iloc[-idx_back_2:], tempos[1][1]+'S', type='bars')
                            data_5 = self.resampler(self.data.iloc[-idx_back_3:], tempos[2][0]+'S', type='bars')
                            data_6 = self.resampler(self.data.iloc[-idx_back_3:], tempos[2][1]+'S', type='bars')

                            pers = periods[1:]
                            for per in pers:
                                per = int(per[4:])
                                data_1['SMA_%d'%per] = self.SMA(data_1.close, per)
                                data_2['SMA_%d'%per] = self.SMA(data_2.close, per)
                                data_3['SMA_%d'%per] = self.SMA(data_3.close, per)
                                data_4['SMA_%d'%per] = self.SMA(data_4.close, per)
                                data_5['SMA_%d'%per] = self.SMA(data_5.close, per)
                                data_6['SMA_%d'%per] = self.SMA(data_6.close, per)

                            data_1_eval = self.add_features(data_1)
                            data_2_eval = self.add_features(data_2)
                            data_3_eval = self.add_features(data_3)
                            data_4_eval = self.add_features(data_4)
                            data_5_eval = self.add_features(data_5)
                            data_6_eval = self.add_features(data_6)

                            data_1_eval.columns = ['%s_%s'%(col,tempos[0][0]) for col in data_1_eval.columns]
                            data_2_eval.columns = ['%s_%s'%(col,tempos[0][1]) for col in data_2_eval.columns]
                            data_3_eval.columns = ['%s_%s'%(col,tempos[1][0]) for col in data_3_eval.columns]
                            data_4_eval.columns = ['%s_%s'%(col,tempos[1][1]) for col in data_4_eval.columns]
                            data_5_eval.columns = ['%s_%s'%(col,tempos[2][0]) for col in data_5_eval.columns]
                            data_6_eval.columns = ['%s_%s'%(col,tempos[2][1]) for col in data_6_eval.columns]

                            model_input_12 = list(data_1_eval.iloc[-1][features_1].values)
                            model_input_12.extend(list(data_2_eval.iloc[-1][features_2].values))
                            model_input_12 = np.reshape(model_input_12, [1, lags, n_features])

                            model_input_34 = list(data_3_eval.iloc[-1][features_3].values)
                            model_input_34.extend(list(data_4_eval.iloc[-1][features_4].values))
                            model_input_34 = np.reshape(model_input_34, [1, lags, n_features])

                            model_input_56 = list(data_5_eval.iloc[-1][features_5].values)
                            model_input_56.extend(list(data_6_eval.iloc[-1][features_6].values))
                            model_input_56 = np.reshape(model_input_56, [1, lags, n_features])
                            
                            prediction_1 = 1 if model_1.predict(model_input_12)[0][0][0] > 0.9 else 0
                            prediction_2 = 1 if model_2.predict(model_input_34)[0][0][0] > 0.9 else 0
                            prediction_3 = 1 if model_3.predict(model_input_56)[0][0][0] > 0.9 else 0
                            self.print('%s %s | %s : %d'%(self.date, self.hour, model_input_12, prediction_1))
                            self.print('%s %s | %s : %d'%(self.date, self.hour, model_input_34, prediction_2))
                            self.print('%s %s | %s : %d'%(self.date, self.hour, model_input_56, prediction_3))
                            allow_entry = True

                            # prediction_1 = 1
                            # prediction_2 = 1
                            # prediction_3 = 1

                        else: allow_entry = False
                        # if pd.to_datetime(self.hour).minute % ana_time != 0 and second > 5:
                        # if second > 5 or pd.to_datetime(self.hour).minute % ana_time != 0:
                        #     allow_entry = False
                            
                        # Entry conditions
                        if not (self.weekday == 4 and pd.to_datetime(self.hour).time() > pd.to_datetime('16:00:00').time()):
                            ## Sells
                            if not sent and prediction_1 > 0 and prediction_2 > 0 and prediction_3 > 0 and allow_entry:
                                max_stop_1 = stop_1*self.leverage*contracts/3
                                max_stop_2 = stop_2*self.leverage*contracts/3
                                max_stop_3 = stop_3*self.leverage*contracts/3
                                price_sell_in_1, sl_sell_1, tp_sell_1, time_sell_in, comm_sell_in_1, profit_sell, ord_sell_sl_1, ord_sell_tp_1 = self.braket_market('SELL', contracts/6, stop_1, target_1, max_stop_1)
                                price_sell_in_2, sl_sell_2, tp_sell_2, time_sell_in, comm_sell_in_2, profit_sell, ord_sell_sl_2, ord_sell_tp_2 = self.braket_market('SELL', contracts/6, stop_1, target_2, max_stop_1, entry_price=price_sell_in_1)     
                                price_sell_in_3, sl_sell_3, tp_sell_3, time_sell_in, comm_sell_in_3, profit_sell, ord_sell_sl_3, ord_sell_tp_3 = self.braket_market('SELL', contracts/6, stop_2, target_3, max_stop_2)
                                price_sell_in_4, sl_sell_4, tp_sell_4, time_sell_in, comm_sell_in_4, profit_sell, ord_sell_sl_4, ord_sell_tp_4 = self.braket_market('SELL', contracts/6, stop_2, target_4, max_stop_2, entry_price=price_sell_in_3)     
                                price_sell_in_5, sl_sell_5, tp_sell_5, time_sell_in, comm_sell_in_5, profit_sell, ord_sell_sl_5, ord_sell_tp_5 = self.braket_market('SELL', contracts/6, stop_3, target_5, max_stop_3)
                                price_sell_in_6, sl_sell_6, tp_sell_6, time_sell_in, comm_sell_in_6, profit_sell, ord_sell_sl_6, ord_sell_tp_6 = self.braket_market('SELL', contracts/6, stop_3, target_6, max_stop_3, entry_price=price_sell_in_5)     
                                if price_sell_in_1 > 0 and price_sell_in_2 > 0 and price_sell_in_3 > 0 and price_sell_in_4 > 0 and price_sell_in_5 > 0 and price_sell_in_6 > 0: sent = True
                                tr_1 = self.x_round(trailing_1 * target_1)
                                tr_2 = self.x_round(trailing_1 * target_2)
                                tr_3 = self.x_round(trailing_2 * target_3)
                                tr_4 = self.x_round(trailing_2 * target_4)
                                tr_5 = self.x_round(trailing_3 * target_5)
                                tr_6 = self.x_round(trailing_3 * target_6)
                                first = False; second = False; third = False; fourth = False; fifth = False; sixth = False

                    # Check for Exit
                    if self.position < 0:
                        try:
                            if not first:  sl_sell_1 = self.trailing_stop(price_in=price_sell_in_1, trailing=tr_1, sl=sl_sell_1, order=ord_sell_sl_1)
                            if not second: sl_sell_2 = self.trailing_stop(price_in=price_sell_in_1, trailing=tr_2, sl=sl_sell_2, order=ord_sell_sl_2)
                            if not third:  sl_sell_3 = self.trailing_stop(price_in=price_sell_in_3, trailing=tr_3, sl=sl_sell_3, order=ord_sell_sl_3)
                            if not fourth: sl_sell_4 = self.trailing_stop(price_in=price_sell_in_3, trailing=tr_4, sl=sl_sell_4, order=ord_sell_sl_4)
                            if not fifth:  sl_sell_5 = self.trailing_stop(price_in=price_sell_in_5, trailing=tr_5, sl=sl_sell_5, order=ord_sell_sl_5)
                            if not sixth:  sl_sell_6 = self.trailing_stop(price_in=price_sell_in_5, trailing=tr_6, sl=sl_sell_6, order=ord_sell_sl_6)
                        except: self.print('Trying to apply trailing stop... Order has been Filled!')

                        # By stop ==========
                        ## Stop 1
                        if self.check_pendings(ord_sell_sl_1) and not first and self.position < 0:   # Check if stop 1 is filled
                            self.exit_pending(ord_sell_sl_1, 'SELL', contracts/6, price_sell_in_1, time_sell_in, comm_sell_in_1, 'sl1')
                            first = True; sent = False
                        ## Stop 2
                        if self.check_pendings(ord_sell_sl_2) and not second and self.position < 0:   # Check if stop 2 is filled
                            self.exit_pending(ord_sell_sl_2, 'SELL', contracts/6, price_sell_in_2, time_sell_in, comm_sell_in_2, 'sl2')
                            second = True; sent = False
                        ## Stop 3
                        if self.check_pendings(ord_sell_sl_3) and not third and self.position < 0:   # Check if stop 3 is filled
                            self.exit_pending(ord_sell_sl_3, 'SELL', contracts/6, price_sell_in_3, time_sell_in, comm_sell_in_3, 'sl3')
                            third = True; sent = False
                        ## Stop 4
                        if self.check_pendings(ord_sell_sl_4) and not fourth and self.position < 0:   # Check if stop 4 is filled
                            self.exit_pending(ord_sell_sl_4, 'SELL', contracts/6, price_sell_in_4, time_sell_in, comm_sell_in_4, 'sl4')
                            fourth = True; sent = False
                        ## Stop 5
                        if self.check_pendings(ord_sell_sl_5) and not fifth and self.position < 0:   # Check if stop 5 is filled
                            self.exit_pending(ord_sell_sl_5, 'SELL', contracts/6, price_sell_in_5, time_sell_in, comm_sell_in_5, 'sl5')
                            fifth = True; sent = False
                        ## Stop 6
                        if self.check_pendings(ord_sell_sl_6) and not sixth and self.position < 0:   # Check if stop 6 is filled
                            self.exit_pending(ord_sell_sl_6, 'SELL', contracts/6, price_sell_in_6, time_sell_in, comm_sell_in_6, 'sl6')
                            sixth = True; sent = False
                        
                        ## False Stop
                        ### Stop 1
                        '''if not self.check_pendings(ord_sell_sl_1) and self.data.iloc[-1].high - sl_sell_1 >= 2 and not first and self.position < 0:
                            self.exit_market(ord_sell_tp_1, 'SELL', contracts/6, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fsl1')
                            first = True; sent = False
                        ### Stop 2
                        if not self.check_pendings(ord_sell_sl_2) and self.data.iloc[-1].high - sl_sell_2 >= 2 and not second and self.position < 0:
                            self.exit_market(ord_sell_tp_2, 'SELL', contracts/6, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fsl2')
                            second = True; sent = False
                        ### Stop 3
                        if not self.check_pendings(ord_sell_sl_3) and self.data.iloc[-1].high - sl_sell_3 >= 2 and not third and self.position < 0:
                            self.exit_market(ord_sell_tp_3, 'SELL', contracts/6, price_sell_in_3, time_sell_in, comm_sell_in_3, 'fsl3')
                            third = True; sent = False
                        ### Stop 4
                        if not self.check_pendings(ord_sell_sl_4) and self.data.iloc[-1].high - sl_sell_4 >= 2 and not fourth and self.position < 0:
                            self.exit_market(ord_sell_tp_4, 'SELL', contracts/6, price_sell_in_4, time_sell_in, comm_sell_in_4, 'fsl4')
                            fourth = True; sent = False
                        ### Stop 5
                        if not self.check_pendings(ord_sell_sl_5) and self.data.iloc[-1].high - sl_sell_5 >= 2 and not fifth and self.position < 0:
                            self.exit_market(ord_sell_tp_5, 'SELL', contracts/6, price_sell_in_5, time_sell_in, comm_sell_in_5, 'fsl5')
                            fifth = True; sent = False
                        ### Stop 6
                        if not self.check_pendings(ord_sell_sl_6) and self.data.iloc[-1].high - sl_sell_6 >= 2 and not sixth and self.position < 0:
                            self.exit_market(ord_sell_tp_6, 'SELL', contracts/6, price_sell_in_6, time_sell_in, comm_sell_in_6, 'fsl6')
                            sixth = True; sent = False'''
                        
                        # By target ==========
                        ## Target 1
                        if self.check_pendings(ord_sell_tp_1) and not first and self.position < 0:    # Check if target 1 is filled
                            self.exit_pending(ord_sell_tp_1, 'SELL', contracts/6, price_sell_in_1, time_sell_in, comm_sell_in_1, 'tp1')
                            first = True; sent = False
                        ## Target 2
                        if self.check_pendings(ord_sell_tp_2) and not second and self.position < 0:    # Check if target 2 is filled
                            self.exit_pending(ord_sell_tp_2, 'SELL', contracts/6, price_sell_in_2, time_sell_in, comm_sell_in_2, 'tp2')
                            second = True; sent = False
                        ## Target 3
                        if self.check_pendings(ord_sell_tp_3) and not third and self.position < 0:    # Check if target 3 is filled
                            self.exit_pending(ord_sell_tp_3, 'SELL', contracts/6, price_sell_in_3, time_sell_in, comm_sell_in_3, 'tp3')
                            third = True; sent = False
                        ## Target 4
                        if self.check_pendings(ord_sell_tp_4) and not fourth and self.position < 0:    # Check if target 4 is filled
                            self.exit_pending(ord_sell_tp_4, 'SELL', contracts/6, price_sell_in_4, time_sell_in, comm_sell_in_4, 'tp4')
                            fourth = True; sent = False
                        ## Target 5
                        if self.check_pendings(ord_sell_tp_5) and not fifth and self.position < 0:    # Check if target 5 is filled
                            self.exit_pending(ord_sell_tp_5, 'SELL', contracts/6, price_sell_in_5, time_sell_in, comm_sell_in_5, 'tp5')
                            fifth = True; sent = False
                        ## Target 6
                        if self.check_pendings(ord_sell_tp_6) and not sixth and self.position < 0:    # Check if target 6 is filled
                            self.exit_pending(ord_sell_tp_6, 'SELL', contracts/6, price_sell_in_6, time_sell_in, comm_sell_in_6, 'tp6')
                            sixth = True; sent = False

                        # Exit on friday
                        if self.weekday == 4 and pd.to_datetime(self.hour).time() >= pd.to_datetime('16:57:00').time() and self.position < 0:
                            if not first:
                                self.exit_market(ord_sell_tp_1, 'SELL', contracts/6, price_sell_in_1, time_sell_in, comm_sell_in_1, 'fri 1')
                                first = True; sent = False
                            if not second:
                                self.exit_market(ord_sell_tp_2, 'SELL', contracts/6, price_sell_in_2, time_sell_in, comm_sell_in_2, 'fri 2')
                                second = True; sent = False
                            if not third:
                                self.exit_market(ord_sell_tp_3, 'SELL', contracts/6, price_sell_in_3, time_sell_in, comm_sell_in_3, 'fri 3')
                                third = True; sent = False
                            if not fourth:
                                self.exit_market(ord_sell_tp_4, 'SELL', contracts/6, price_sell_in_4, time_sell_in, comm_sell_in_4, 'fri 4')
                                fourth = True; sent = False
                            if not fifth:
                                self.exit_market(ord_sell_tp_5, 'SELL', contracts/6, price_sell_in_5, time_sell_in, comm_sell_in_5, 'fri 5')
                                fifth = True; sent = False
                            if not sixth:
                                self.exit_market(ord_sell_tp_6, 'SELL', contracts/6, price_sell_in_6, time_sell_in, comm_sell_in_6, 'fri 6')
                                sixth = True; sent = False

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
                        except:
                            self.print('Connection Failed! Trying to reconnect in 10 seconds...')

            self.ib.disconnect()
            self.print('%s %s | Session Ended. Good Bye!' % (self.date, self.hour))

if __name__ == '__main__':
    symbol = 'MES'
    port = 7497
    client = 253

    live_octopus = LiveOctopus(symbol=symbol, bot_name='Octopus Shorts (demo)', temp='1 min', port=port, client=client, real=False)
    
    init = ['2022-04-15', '2022-03-25', '2022-01-28']
    final = '2022-04-22'
    
    periods = ['close', 'SMA_21', 'SMA_89']
    #tempos = ['540', '720']          #['180', '240'] ['540', '720']
    tempos = [['60', '120'], ['180', '240'], ['540', '720']]
    live_octopus.run_strategy(contracts=6, stop_1=3, target_1=2, target_2=3, trailing_1=0.9, stop_2=6, target_3=3, target_4=10, trailing_2=0.9,
                          stop_3=18, target_5=11, target_6=19, trailing_3=0.8, periods=periods, tempos=tempos, init=init, final=final)