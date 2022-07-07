import pandas as pd
# import quandl
import pandas_datareader as web
import datetime
import os 
import matplotlib
# 后台执行，如果需要使用本地运行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import base64
from io import BytesIO
import scipy.optimize as sco

class Markowitz:
    def __init__(self, selected , titles, debt_name, start_time, end_time):
        self.selected = selected;
        self.titles = titles;
        self.risk_free_national_debt = debt_name;
        self.start_time = start_time;
        self.end_time = end_time;
        self.spc_adjclose =  self.month_change("S&P 500" , "^GSPC", self.start_time, self.end_time);
        self.table = self.getData();


    def get_risk_free_interest_rate(self, debt_name, start, end):
        df = web.DataReader(debt_name,'yahoo', start, end);
        returns_annual = df.mean()/100;
        return returns_annual['Adj Close'];

    def get_last_risk_free_interest_rate(self, debt_name, start, end):
        df = web.DataReader(debt_name,'yahoo', start, end);
        interest = (df.tail(1)["Adj Close"].values[0])/100
        return interest;

    def month_change(self, title, code, start, end):
        data = web.DataReader(code,'yahoo',start, end)
        pct_change = data.resample('1M').mean().pct_change().dropna();
        print("{start_time_str}——{end_time_str}{stock_title}月收益".format(stock_title=title, start_time_str=self.start_time.strftime("%Y年%m月%d日"), end_time_str=self.end_time.strftime("%Y年%m月%d日")));
        print(pct_change);
        adjclose = np.array(pct_change["Adj Close"])
        # plt.title(title);
        # adjclose["Adj Close"].plot();
        # plt.show();
        return adjclose;

    def getMonthChanges(self):
        i=0;
        datas = [];
        for code in self.selected:
            data = self.month_change(self.titles[i], code, self.start_time,  self.end_time);
            datas.append(data);
            i += 1;
        return datas;


    def getData(self):
        data = web.DataReader(self.selected,'yahoo', self.start_time, self.end_time);
        data = data.stack(level = 1).reset_index(level = [0, 1], drop = False).reset_index();
        df = data[['Date','Adj Close','Symbols']];
        data = df.rename(columns={'Date':'date', 'Symbols':'ticker', 'Adj Close':'adj_close'})

        clean = data.set_index('date');
        table = clean.pivot(columns='ticker');
        
        return table;

    def statistics(self, weights):    
        
        #根据权重，计算资产组合收益率/波动率/夏普率。
        #输入参数
        #==========
        #weights : array-like 权重数组
        #权重为股票组合中不同股票的权重    
        #返回值
        #=======
        #pret : float
        #      投资组合收益率
        #pvol : float
        #      投资组合波动率
        #pret / pvol : float
        #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产
        #
        table = self.table;
        returns_daily = table.pct_change();
        returns_annual = returns_daily.mean() * 252;
        weights = np.array(weights)
        pret = returns = np.dot(weights, returns_annual);
        pvol = np.sqrt(np.dot(weights.T, np.dot(returns_daily.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])

    def min_func_sharpe(self, weights):
        return -self.statistics(weights)[2]

    def getSharpBySLSQP(self):
        table = self.table;
        number_of_assets = len(self.selected);
        bnds = tuple((0, 1) for x in range(number_of_assets));
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1});
        opts = sco.minimize(self.min_func_sharpe, number_of_assets * [1. / number_of_assets,], method='SLSQP',  bounds=bnds, constraints=cons)
        return opts;
    
    def getPricePlot(self):
        table = self.table;
        plt.figure(figsize=(14, 7))
        for c in table.columns.values:
            plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c[1])
        plt.legend(loc='upper left', fontsize=12)
        plt.ylabel('price in $')
        # plt.show();

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
        plot_data = buffer.getvalue()
        imb = base64.encodebytes(plot_data,)   
        ims = imb.decode()
        plt.close()
        return ims;

    def getModels(self, pct_change_datas):
        i = 0
        models = [];
        for code in self.selected:
            adjclose = pct_change_datas[0];
            if (adjclose.size == self.spc_adjclose.size):
                X = sm.add_constant(self.spc_adjclose);
                model = sm.OLS(adjclose, X);
                fit = model.fit();
                # print(fit.summary());
                # print(fit.params);
                models.append({"title":self.titles[i], "fit":fit});
                # print(fit.rsquared)
            else:
                print("数组长度不一样");
            i += 1;
        return models;

    

    def getPortfolio(self):
        table = self.table;
        returns_daily = table.pct_change();
        returns_annual = returns_daily.mean() * 252;
        cov_daily = returns_daily.cov();
        cov_annual = cov_daily * 252;

        port_returns = [];
        port_volatility = [];
        sharpe_ratio = [];
        stock_weights = [];

        num_assets = len(self.selected);
        num_portfolios = 50000;

        np.random.seed(101);

        risk_free_interest_rate  = self.get_last_risk_free_interest_rate(self.risk_free_national_debt, self.start_time, self.end_time);

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

        for counter,symbol in enumerate(self.selected):
            portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights];

        df = pd.DataFrame(portfolio);

        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in self.selected];

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
        # plt.show();
     
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
        plot_data = buffer.getvalue()
        imb = base64.encodebytes(plot_data,)   
        ims = imb.decode()
        plt.close()
        print(min_variance_port.T);
        print(sharpe_portfolio.T);
        return {"min_variance_port":min_variance_port.T,"sharpe_portfolio":sharpe_portfolio.T,"ims":ims};
    
    





