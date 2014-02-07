import random, itertools, math, os, json, urllib2, sys, copy

from datetime import date, datetime
import time as timer
import numpy as np
import multiprocessing as mp

def readable_date(unix):
	return datetime.fromtimestamp(int(unix)).strftime('%Y-%m-%d %H:%M:%S')

class Indicator(object):
	def __init__(self):
		raise NotImplementedError("Must be implemented on per-indicator basis")

	def tick(self, tick, past):
		self.process(tick, past)
		self.days += 1

	def process(self, data):
		raise NotImplementedError("Must be implemented on per-indicator basis")

	def get(self):
		return self.values

	def x_axis(self):
		return self.x_axis

class SMA(Indicator):
	def __init__(self, period):
		self.period = period
		self.days = 0
		self.values = []
		self.x_axis = []
		self.under = False
		self.over = False
		self.crossed_over = False
		self.crossed_under = False

	def process(self, tick, past):
		if self.days - 1 == self.period:
			sum = 0
			for day in past:
				sum += day['close']
			self.values.append(sum / len(past))
			self.x_axis.append(self.days)
		elif self.days - 1 > self.period:
			self.crossed_over = self.crossed_under = False
			sma = self.values[-1] + ((tick['close'] - self.values[-1]) / self.period)
			self.values.append(sma)
			self.x_axis.append(self.days)

			if tick['close'] > sma and not self.over:
				self.crossed_over = True
				self.over = True
				self.under = False
			elif tick['close'] < sma and not self.under:
				self.crossed_under = True
				self.under = True
				self.over = False

class EMA(Indicator):
	def __init__(self, period):
		self.period = period
		self.days = 0
		self.values = []
		self.x_axis = []

	def process(self, tick, past):
		if self.days - 1 == self.period:
			sum = 0
			for day in past:
				sum += day['close']
			self.values.append(sum / len(past))
			self.x_axis.append(self.days)
		elif self.days - 1 > self.period:
			multiplier = 2 / (float(self.period) + 1)
			self.values.append(((tick['close'] - self.values[-1]) * multiplier) + self.values[-1])
			self.x_axis.append(self.days)

class RSI(Indicator):
	def __init__(self, period):
		self.period = period
		self.days = 0
		self.values = []
		self.x_axis = []

		self.ema = []
		self.ema_x_axis = []
		self.sma = 0
		self.avg_gains = []
		self.avg_losses = []

		self.conditions_x = []
		self.overbought = []
		self.oversold = []

		self.is_oversold = False
		self.is_overbought = False

	def process(self, tick, past):
		if self.days - 1 == self.period:
			losses = 0
			gains = 0
			prev_price = past[-1]['close']

			for price in past[1:]:
				price = price['close']
				price_change = price - prev_price
				if price_change > 0:
					gains += price_change
				else:
					losses += abs(price_change)
				prev_price = price

			self.avg_gains.append(gains / self.period)
			self.avg_losses.append(losses / self.period)
		elif self.days - 1 > self.period:
			price_change = tick['close'] - past[-2]['close']

			gain = price_change
			gain = gain if gain > 0 else 0

			loss = price_change
			loss = abs(loss) if loss < 0 else 0

			avg_gain = ((self.avg_gains[-1] * (self.period - 1)) + gain) / self.period
			self.avg_gains.append(avg_gain)

			avg_loss = ((self.avg_losses[-1] * (self.period - 1)) + loss) / self.period
			self.avg_losses.append(avg_loss)

			if avg_loss == 0:
				avg_loss = 1

			rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
			self.values.append(rsi)
			self.x_axis.append(self.days)

			if len(self.values) == self.period:
				self.sma = sum(self.values) / len(self.values)
			elif len(self.values) > self.period:
				self.sma = self.sma + ((self.values[-1] - self.values[-2]) / self.period)
				self.ema_x_axis.append(self.days)

				# for the first ema calculation the ema is equal to the sma
				if len(self.ema) == 0:
					self.ema.append(self.sma)
				else:
					multiplier = 2 / (float(self.period) + 1)
					self.ema.append(((self.values[-1] - self.ema[-1]) * multiplier) + self.ema[-1])

		if len(self.ema) > 2:
			rsi_avg = sum(self.ema[:-1]) / len(self.ema[:-1])
			variance = 0
			i = 0
			for rsi in self.ema[:-1]:
				variance += math.pow(rsi - rsi_avg, 2)
				i += 1
			stddev = math.sqrt(variance / (i - 1))

			self.conditions_x.append(self.days)
			overbought_level = rsi_avg + 2 * stddev
			oversold_level = rsi_avg - 2 * stddev
			self.is_overbought = self.values[-1] > overbought_level
			self.is_oversold = self.values[-1] < oversold_level
			self.overbought.append(overbought_level)
			self.oversold.append(oversold_level)

class Volume(Indicator):
	def __init__(self):
		self.values = []
		self.x_axis = []
		self.colors = []
		self.days = 0
		self.ema_period = 20
		self.ema = 0
		self.ema_values = []
		self.ema_x_axis = []

	def process(self, tick, past):
		self.x_axis.append(self.days)
		self.values.append(tick['volume'])
		if tick['close'] > tick['open']:
			self.colors.append('g')
		else:
			self.colors.append('r')

		if self.days + 1 == self.ema_period:
				self.sma = sum(self.values) / len(self.values)
				self.ema_values.append(self.sma)
				self.ema_x_axis.append(self.days)
		elif self.days + 1 > self.ema_period:
			self.sma = self.sma + ((self.values[-1] - self.values[-2]) / self.ema_period)
			self.ema_x_axis.append(self.days)

			multiplier = 2 / (float(self.ema_period) + 1)
			self.ema_values.append(((self.values[-1] - self.ema_values[-1]) * multiplier) + self.ema_values[-1])

class SupportResistance(Indicator):
	def __init__(self, move_size):
		self.lows = []
		self.highs = []
		self.start_price = 0
		self.move_size = move_size
		self.days = 0

		self.hit_low = False
		self.hit_high = False

		self.new_high = False
		self.new_low = False

	def process(self, tick, past):
		if self.days == 0:
			self.highs.append([self.days, tick['close']])
			self.lows.append([self.days, tick['close']])
		else:
			self.new_high = self.new_low = False
			if not self.hit_low and tick['high'] > self.highs[-1][1]:
				self.highs[-1] = ([self.days, tick['high']])
			elif not self.hit_high and tick['low'] < self.lows[-1][1]:
				self.lows[-1] = ([self.days, tick['low']])
			elif not self.hit_high and (tick['high'] * (1 - self.move_size)) > self.lows[-1][1]:
				self.highs.append([self.days, tick['high']])
				self.hit_high = True
				self.new_high = True
				self.hit_low = False
			elif not self.hit_low and tick['low'] < (self.highs[-1][1] * (1 - self.move_size)):
				self.lows.append([self.days, tick['low']])
				self.hit_high = False
				self.new_low = True
				self.hit_low = True

	def higher_high(self):
		if len(self.highs) < 2:
			return False
		return self.highs[-1][1] > self.highs[-2][1]

	def lower_high(self):
		if len(self.highs) < 2:
			return False
		return self.highs[-1][1] < self.highs[-2][1]

	def higher_low(self):
		if len(self.lows) < 2:
			return False
		return self.lows[-1][1] > self.lows[-2][1]

	def lower_low(self):
		if len(self.lows) < 2:
			return False
		return self.lows[-1][1] < self.lows[-2][1]

	def get(self):
		return {
			'highs': self.highs,
			'lows': self.lows
		}

class Agent(object):
	def __init__(self, indicators, instructions, portfolio, ticker):
		self.indicators = indicators
		self.instructions = instructions
		self.ticker = ticker
		self.portfolio = portfolio
		self.ticks = []
		self.num_ticks = 0

	def tick(self, data):
		self.num_ticks += 1
		self.ticks.append(data)
		for indicator in self.indicators:
			indicator['instance'].tick(data, self.ticks)
		self.process_instructions(data)

	def process_instructions(self, data):
		for instruction in self.instructions:
			execute = True
			for conditional in instruction['conditions']:
				execute = execute and conditional.check(self.indicators)
			if execute:
				if self.portfolio.in_long and instruction['in_long'] == 'cover':
					self.portfolio.close_long(self.ticker, data['close'], self.num_ticks)
				if self.portfolio.in_short and instruction['in_short'] == 'cover':
					self.portfolio.close_short(self.ticker, data['close'], self.num_ticks)

				if not self.portfolio.in_long and instruction['no_position'] == 'long':
					self.portfolio.go_long(self.ticker, data['close'], self.num_ticks)
				if not self.portfolio.in_short and instruction['no_position'] == 'short':
					self.portfolio.go_short(self.ticker, data['close'], self.num_ticks)

class Portfolio(object):
	def __init__(self, balance, position_size):
		self.balance = balance
		self.position_size = position_size
		self.in_long = False
		self.in_short = False
		self.positions = []
		self.leverage = 1

	def go_long(self, ticker, price, tick):
		if self.in_long or self.in_short:
			return False
		shares = int(self.position_size / price)
		self.balance -= price * shares * self.leverage
		self.positions.append({
			'ticker': ticker, 'entry_price': price, 'shares': shares, 'type': 'long', 'open': True, 'tick_open': tick
		})
		self.in_long = True
		self.position_size = self.balance

	def go_short(self, ticker, price, tick):
		if self.in_short or self.in_long:
			return False
		shares = int(self.position_size / price)
		self.balance += price * shares * self.leverage
		self.positions.append({
			'ticker': ticker, 'entry_price': price, 'shares': shares, 'type': 'short', 'open': True, 'tick_open': tick
		})
		self.in_short = True
		self.position_size = self.balance

	def close_long(self, ticker, price, tick):
		if not self.in_long:
			return False
		for pos in self.positions:
			if pos['ticker'] == ticker and pos['type'] == 'long' and pos['open'] == True:
				self.balance += pos['shares'] * price * self.leverage
				pos['open'] = False
				pos['exit_price'] = price
				pos['tick_close'] = tick
				pos['net'] = pos['shares'] * (price - pos['entry_price'])
				self.in_long = False
				self.position_size = self.balance
				return True
		return False

	def close_short(self, ticker, price, tick):
		if not self.in_short:
			return False
		for pos in self.positions:
			if pos['ticker'] == ticker and pos['type'] == 'short' and pos['open'] == True:
				self.balance -= pos['shares'] * price * self.leverage
				pos['open'] = False
				pos['exit_price'] = price
				pos['tick_close'] = tick
				pos['net'] = pos['shares'] * (pos['entry_price'] - price)
				self.in_short = False
				self.position_size = self.balance
				return True
		return False

	def close_all(self, ticker, price, tick):
		return self.close_long(ticker, price, tick) or self.close_short(ticker, price, tick)

class Comparator(object):
	def __init__(self, indicator, comparison, rightside, english):
		indicator = indicator.split('.')
		self.indicator = indicator
		self.comparison = comparison
		self.rightside = rightside
		self.english = english

	def check(self, indicators):
		indicator_name = self.indicator[0]
		indicator_method = self.indicator[1]
		indicator_instance = False
		for indicator in indicators:
			if indicator['name'] == indicator_name:
				indicator_instance = indicator['instance']
				break
		if indicator_instance:
			if indicator_method[-2:] != "()":
				result = getattr(indicator_instance, indicator_method)
			else:
				result = getattr(indicator_instance, indicator_method[:-2])()
			expression = "%s %s %s" % (result, self.comparison, self.rightside)
			return eval(expression)

class DataReader(object):
	def __init__(self, ticker):
		raise NotImplementedError("Must be implemented for each data reader")

	def __iter__(self):
		return self

	def next(self):
		try:
			self.counter += 1
			return self.format(self.data[self.counter - 1])
		except IndexError:
			raise StopIteration

	def format(self):
		raise NotImplementedError("Must be implemented for each data reader")

	def store(self):
		raise NotImplementedError("Must be implemented for each data reader")

	def retrieve(self, key):
		raise NotImplementedError("Must be implemented for each data reader")

	def exists(self, key):
		return False

	def getstart(self):
		return self.starttime

class GoogleReader(DataReader):
	def __init__(self, ticker, interval, period):
		self.counter = 0
		self.key = "%s,%s,%s" % (ticker, interval, period)
		path = 'data/%s-%s.txt' % (ticker, self.key)

		try:
		   open(path)
		except IOError:
			try:
				url = "https://www.google.com/finance/getprices?q=%s&i=%s&p=%sd&f=d,o,h,l,c,v" % (ticker, interval, period)
				u = urllib2.urlopen(url)
				local = open(path, 'w')
				local.write(u.read())
				local.close()
			except Exception:
				print "%s not found" % ticker
				sys.exit()
			
		self.data = open(path, 'r')
		self.data = map(str.strip, self.data.readlines())

		self.columns = {
			'close': 1,
			'high': 2,
			'low': 3,
			'open': 4,
			'volume': 5
		}

		self.rows = {
			'date': 7,
			'ticks': 8
		}

		for line in self.data:
			if self.counter == self.rows['date']:
				self.starttime = line.split(',')[0][1:]
			if self.counter == self.rows['ticks']:
				self.formatted_data = self.format(line)
				break;
			self.counter += 1

	def format(self, data):
		data = data.split(',')
		self.counter += 1
		return {
			'row_number': self.counter,
			'close': float(data[self.columns['close']]),
			'high': float(data[self.columns['high']]),
			'low': float(data[self.columns['low']]),
			'open': float(data[self.columns['open']]),
			'volume': float(data[self.columns['volume']])
		}

instructions = [
	{
		'in_long': 'cover',
		'in_short': 'hold',
		'no_position': 'hold',
		'conditions': [
			Comparator('rsi.is_oversold', '==', True, 'RSI signals oversold conditions'),
			Comparator('supportresistance.lower_high()', '==', True, 'Lower high'),
			Comparator('supportresistance.lower_low()', '==', True, 'Lower low')
		]
	},
	{
		'in_long': 'hold',
		'in_short': 'cover',
		'no_position': 'hold',
		'conditions': [
			Comparator('supportresistance.higher_high()', '==', True, 'Higher high'),
			Comparator('supportresistance.higher_low()', '==', True, 'Higher low')
		]
	},
	{
		'in_long': 'hold',
		'in_short': 'hold',
		'no_position': 'short',
		'conditions': [
			Comparator('rsi.is_oversold', '==', True, 'RSI signals overbought conditions'),
			Comparator('supportresistance.lower_high()', '==', True, 'Lower high'),
			Comparator('supportresistance.lower_low()', '==', True, 'Lower low')
		]
	},
	{
		'in_long': 'hold',
		'in_short': 'hold',
		'no_position': 'long',
		'conditions': [
			Comparator('rsi.is_oversold', '==', True, 'RSI signals oversold conditions'),
			Comparator('supportresistance.higher_high()', '==', True, 'Higher high'),
			Comparator('supportresistance.higher_low()', '==', True, 'Higher low')
		]
	}
]

indicators = [
	{
		'label': '20 Period SMA',
		'instance': SMA(20),
		'name': 'sma'
	},
	{
		'label': '20 Period EMA',
		'instance': EMA(20),
		'name': 'ema'
	},
	{
		'label': '14 Period RSI',
		'instance': RSI(14),
		'name': 'rsi'
	},
	{
		'label': 'Volume',
		'instance': Volume(),
		'name': 'volume'
	},
	{
		'label': 'Support and Resistance',
		'instance': SupportResistance(.01),
		'name': 'supportresistance'
	}
]

tickers = open('tickers.txt', 'r')
tickers = random.sample([line.split(',') for line in tickers.readlines()][0], 3)
print tickers

def analyze(tickers, portfolio):
	holdings = []
	initial_balance = portfolio.balance

	for ticker in tickers:
		holdings.append({
			'agent': Agent(copy.deepcopy(indicators), copy.deepcopy(instructions), portfolio, ticker),
			'reader': GoogleReader(ticker, interval=5, period=1),
			'ticker': ticker
		})

	i = 0
	do_continue = True
	base_gains = []
	while do_continue:
		for holding in holdings:
			try:
				tick = holding['reader'].next()
				if i == 0:
					startprice = float(tick['open'])
				holding['agent'].tick(tick)
				holding['price'] = tick['close']
			except StopIteration:
				endprice = float(tick['close'])
				base_gains.append((endprice - startprice) / startprice * 100)
				do_continue = False
				break

	for holding in holdings:
		portfolio.close_all(holding['ticker'], holding['price'], i)

	print "Base Gain: %.02f%%" % (sum(base_gains) / len(base_gains))
	print "Algo Gain: %.02f%%" % ((portfolio.balance - initial_balance) / initial_balance * 100)
	print "Beginning Balance: $%.02f" % initial_balance
	print "Ending Balance: $%.02f" % portfolio.balance

	# print json.dumps(portfolio.positions, indent=4)

	plot = True
	plot_extra = True
	plot_positions = True
	if plot:
		import matplotlib.pyplot as plt
		from matplotlib.finance import candlestick, candlestick2

		for holding in holdings:
			candledata = []
			time = 0

			for tick in holding['agent'].ticks:
				candledata.append([float(time), tick['open'], tick['close'], tick['low'], tick['high']])
				time += 1

			fig, ax = plt.subplots()

			ax = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
			ax.spines['right'].set_color('none')
			ax.spines['top'].set_color('none')
			ax.yaxis.set_ticks_position('left')
			ax.spines['left'].set_linewidth(2)
			ax.spines['bottom'].set_linewidth(2)
			ax.set_xlim(-1, time)
			plt.title(holding['ticker'])
			plt.setp(ax.get_xticklabels(), visible=False)

			fig.subplots_adjust(bottom=0.2)
			candlestick(ax, candledata, width=0.6, colorup='g', colordown='r')

			# TODO: make plotting indicators less procedural
			if plot_extra:
				for indicator in holding['agent'].indicators:
					name = indicator['label']
					indicator = indicator['instance']
					if name in ['20 Period SMA', '20 Period EMA']:
						ax.plot(indicator.x_axis, indicator.values, label=name)
					elif False and name == '14 Period RSI':
						ax2 = ax.twinx()
						ax2.set_xlim(-1, time)
						ax2.plot(indicator.x_axis, indicator.values)
						ax2.plot(indicator.conditions_x, indicator.overbought, ls='--')
						ax2.plot(indicator.conditions_x, indicator.oversold, ls='--')
					elif name == 'Volume':
						bottom = plt.subplot2grid((4,4), (3,0), rowspan=2, colspan=4)
						bottom.set_xlim(-.5, time)
						bottom.bar(indicator.x_axis, indicator.values, width=0.6, color=indicator.colors)
						bottom.plot(indicator.ema_x_axis, indicator.ema_values)
					elif name == 'Support and Resistance':
						ax.plot(*np.transpose(indicator.highs), marker='', color='y', ls='--')
						ax.plot(*np.transpose(indicator.lows), marker='', color='y', ls='--')
			if plot_positions:
				for position in portfolio.positions:
					if position['ticker'] == holding['ticker']:
						# plot blue dots for long positions
						if position['type'] == 'long':
							marker = 'ob'
						# plot red dots for short positions
						elif position['type'] == 'short':
							marker = 'or'
						ax.plot(position['tick_open'], position['entry_price'], marker)

						'''
						ax.annotate('cover', xy=(position['tick_close'], position['exit_price']),  xycoords='data',
				   			xytext=(-30, -30), textcoords='offset points',
				   			arrowprops=dict(arrowstyle="->",
				   			connectionstyle="arc3,rad=.2")
			   			)
						'''

						# plot yellow dots when the position is closed (short or long)
						ax.plot(position['tick_close'], position['exit_price'], 'oy')

			plt.legend()
			filename = './graphs/%s.png' % holding['ticker']
			plt.savefig(filename,dpi=300)

	return {'baseline': float((endprice - startprice) / endprice) * 100,
			'algo': float(((float(portfolio.balance) - initial_balance) / initial_balance)) * 100,
			'num_ticks': i}
	return "Finished Process #%d" % (os.getpid())

def log_result(result):
	if result != 0:
		results['algo'].append(result['algo'])
		results['baseline'].append(result['baseline'])

def go():
	start = timer.time()

	initial_balance = 10000
	position_size = initial_balance / len(tickers)
	multithreaded = False

	if multithreaded:
		pool = mp.Pool(4)
		for ticker in tickers:
			portfolio = Portfolio(initial_balance)
			pool.apply_async(analyze, args=(ticker,portfolio,), callback=log_result)
		pool.close()
		pool.join()
	else:
		portfolio = Portfolio(initial_balance, position_size)
		result = analyze(tickers, portfolio)

	print "Elapsed Time: %f" % (timer.time() - start)

if __name__ == '__main__':
	go()