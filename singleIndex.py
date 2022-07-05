from operator import mod
from pickle import TRUE
import pandas as pd
# import quandl
import pandas_datareader as web
import datetime
import os 
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

selected = ['LI', 'NIO', 'XPEV'];
titles = ["小鹏", '蔚来', '理想'];
risk_free_national_debt = '^FVX';

start_time = datetime.datetime(2020,9,1);
end_time = datetime.datetime(2022,7,1);


def get_risk_free_interest_rate(debt_name, start, end):
    df = web.DataReader(debt_name,'yahoo', start, end);
    returns_annual = df.mean()/100;
    return returns_annual['Adj Close'];


def get_last_risk_free_interest_rate(debt_name, start, end):
    df = web.DataReader(debt_name,'yahoo', start, end);
    interest = (df.tail(1)["Adj Close"].values[0])/100
    return interest;

def month_change(title, code, start, end):
    data = web.DataReader(code,'yahoo',start, end)
    pct_change = data.resample('1M').mean().pct_change().dropna();
    print("{start_time_str}——{end_time_str}{stock_title}月收益".format(stock_title=title, start_time_str=start_time.strftime("%Y年%m月%d日"), end_time_str=end_time.strftime("%Y年%m月%d日")));
    print(pct_change);
    adjclose = np.array(pct_change["Adj Close"])
    # plt.title(title);
    # adjclose["Adj Close"].plot();
    # plt.show();
    return adjclose;
i = 0

spc_adjclose = month_change("S&P 500" , "^GSPC", start_time, end_time);

for code in selected:
    adjclose = month_change(titles[i], code, start_time,  end_time);
    if (adjclose.size == spc_adjclose.size):
        X = sm.add_constant(spc_adjclose);
        model = sm.OLS(adjclose, X);
        fit = model.fit();
        # print(fit.summary());
        # print(fit.params);
        print("得到{title}模型为 R1 = {const} + {x1}Rm + e1".format(title=titles[i], const=fit.params[0],x1=fit.params[1]));
        # print(fit.rsquared)
    else:
        print("数组长度不一样");
    i += 1;


data = web.DataReader(selected,'yahoo', start_time, end_time);

data = data.stack(level = 1).reset_index(level = [0, 1], drop = False).reset_index();

df = data[['Date','Adj Close','Symbols']];

data = df.rename(columns={'Date':'date', 'Symbols':'ticker', 'Adj Close':'adj_close'})

clean = data.set_index('date');
table = clean.pivot(columns='ticker');
returns_daily = table.pct_change();
returns_annual = returns_daily.mean() * 250;
cov_daily = returns_daily.cov();
cov_annual = cov_daily * 250;

port_returns = [];
port_volatility = [];
sharpe_ratio = [];
stock_weights = [];

num_assets = len(selected);
num_portfolios = 50000;

np.random.seed(101);

risk_free_interest_rate  = get_last_risk_free_interest_rate(risk_free_national_debt, start_time, end_time);

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets);
    weights /= np.sum(weights);
    returns = np.dot(weights, returns_annual);
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)));
    sharpe = (returns - risk_free_interest_rate)/ volatility;
    sharpe_ratio.append(sharpe);
    port_returns.append(returns);
    port_volatility.append(volatility);
    stock_weights.append(weights);

portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio};

for counter,symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights];

df = pd.DataFrame(portfolio);

column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected];

df = df[column_order];

# plt.style.use('seaborn-dark')
# df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
#                 cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
# plt.xlabel('Volatility (Std. Deviation)')
# plt.ylabel('Expected Returns')
# plt.title('Efficient Frontier')
# plt.show()

min_volatility = df['Volatility'].min();
max_sharpe = df['Sharpe Ratio'].max();

sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe];
min_variance_port = df.loc[df['Volatility'] == min_volatility];

plt.style.use('seaborn-dark');
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True);
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200);
x = list(sharpe_portfolio['Volatility'])[0];
y = list(sharpe_portfolio['Returns'])[0];

plt.plot([0, x] , [risk_free_interest_rate, y] ,color = 'r')
plt.text(np.round(x,4)+0.01,np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10, color="red")
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
x = list(min_variance_port['Volatility'])[0];
y = list(min_variance_port['Returns'])[0];
plt.text(np.round(x,4)+0.01,np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10, color="white");
plt.xlabel('Volatility (Std. Deviation)');
plt.ylabel('Expected Returns');
plt.title('Efficient Frontier');
print(min_variance_port.T);
print(sharpe_portfolio.T);

plt.show();






