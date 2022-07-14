from operator import rshift
import pandas as pd
# import quandl
import pandas_datareader as web
import datetime
import os 
import matplotlib
# 后台执行，如果需要使用本地运行

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import base64
from io import BytesIO
import scipy.optimize as sco
import scipy.interpolate as sci


class Markowitz:
    def __init__(self, selected , debt_name, start_time, end_time, datasource):
        self.datasource = datasource;
        self.selected = selected;
        self.risk_free_national_debt = debt_name;
        self.start_time = start_time;
        self.end_time = end_time;
        self.datasource.setSelected(selected);
        self.spc_adjclose =  self.datasource.getMonthStardardData();
        self.table = self.getData();

    def get_risk_free_interest_rate(self, debt_name, start, end):
        df = web.DataReader(debt_name,'yahoo', start, end);
        returns_annual = df.mean()/100;
        return returns_annual['Adj Close'];

    def get_last_risk_free_interest_rate(self, debt_name, start, end):
        df = web.DataReader(debt_name,'yahoo', start, end);
        interest = (df.tail(1)["Adj Close"].values[0])/100
        return interest;

    def getMonthChanges(self):
        i=0;
        datas = [];
        for code in self.selected:
            data = self.datasource.getMonthDataByCode(code);
            datas.append(data);
            i += 1;
        return datas;


    def getData(self):
        data = self.datasource.getDatas();
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
            adjclose = pct_change_datas[i];
            if (adjclose.size == self.spc_adjclose.size):
                X = sm.add_constant(self.spc_adjclose);
                model = sm.OLS(adjclose, X);
                fit = model.fit();
                print(fit.summary());
                # print(fit.params);
                models.append({"title":code, "fit":fit});
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
        plt.text(np.round(x,4)+0.01,np.round(y,4),(np.round(x,4),np.round(y,4)),ha='left',va='bottom',fontsize=10, color="red")

        plt.plot([0, x] , [risk_free_interest_rate, y] ,color = 'r')

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
        return {"min_variance_port":min_variance_port.T,"sharpe_portfolio":sharpe_portfolio.T,"ims":ims,"df":df};
    

    def min_func_portfolio(self, weights):
        return self.statistics(weights)[1]

    def getTangentPoint(self):
        num_assets = len(self.selected);
        weights = np.random.random(num_assets);
        weights /= np.sum(weights)
        bnds = tuple((0, 1) for x in weights)

        target_sample = np.linspace(0.0, 1, 50)
        target_returns = [];
        target_volatilities = []
        i = 0;
        for tret in target_sample:
            cons = ({'type': 'eq', 'fun': lambda x:  self.statistics(x)[0] - tret},
                    {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
            res = sco.minimize(self.min_func_portfolio, num_assets * [1. / num_assets,], method = 'SLSQP',
                            bounds = bnds, constraints = cons)
            tfun = round(res['fun'],8)
          
            if len(target_volatilities) > 0 and target_volatilities[-1] == tfun :
                print('--> TVOLS dupl value={value}'.format(value=target_volatilities[-1]))
                i += 1;
                continue
            target_volatilities.append(tfun)
            target_returns.append(target_sample[i]);
            i += 1;
        target_returns = np.array(target_returns);
        target_volatilities = np.array(target_volatilities)

        ind = np.argmin(target_volatilities)        # returns index of smallest element  
        efficient_volatilities = target_volatilities[ind:]       # takes values greater than the min variance
        efficient_returns = target_returns[ind:]                                    ####
        efficient_volatilities, efficient_returns = (list(t) for t in zip(*sorted(zip(efficient_volatilities, efficient_returns))))

        tck = sci.splrep(efficient_volatilities, efficient_returns)     # BSpline object representation
        # tck is a tuple (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.

        def f(x):
            ''' Efficient frontier function (splines approximation). '''
            return sci.splev(x, tck, der = 0)         # evaluate a BSpline
        def df(x):
            ''' First derivative of efficient frontier function. '''
            return sci.splev(x, tck, der = 1)         # evaluate a BSpline with first derivation


        def equations(p, rf = 0.03):
            eq1 = rf - p[0]
            eq2 = rf + p[1] * p[2] - f(p[2])
            eq3 = p[1] - df(p[2])
            return eq1, eq2, eq3

        risk_free_interest_rate  = self.get_last_risk_free_interest_rate(self.risk_free_national_debt, self.start_time, self.end_time);
        opt = sco.fsolve(equations, [0.3, 0, 0.7], [risk_free_interest_rate])
        print(opt)
        result = np.round(equations(opt, risk_free_interest_rate), 6)
        print(result)
        return [result, opt, target_volatilities, target_returns]
    def getTargetPlot(self, target_volatilities, target_returns, df):
        #画散点图
        plt.style.use('seaborn-dark');
        plt.figure(figsize=(9, 5))
        df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True);
        #圆点为随机资产组合
        # plt.scatter(portfolio_volatilities, portfolio_returns,
        #             c=portfolio_returns / portfolio_volatilities, marker='o')
        #叉叉为有效边界            
        plt.scatter(target_volatilities, target_returns,
                    c=target_returns / target_volatilities, marker='x')
        # #红星为夏普率最大值的资产组合            
        # plt.plot(self.statistics(opts['x'])[1], self.statistics(opts['x'])[0],
        #         'r*', markersize=15.0)
        # #黄星为最小方差的资产组合            
        # plt.plot(self.statistics(optv['x'])[1], self.statistics(optv['x'])[0],
        #         'y*', markersize=15.0)
        #             # minimum variance portfolio
        plt.grid(True)
        plt.xlabel('Volatility (Std. Deviation)');
        plt.ylabel('Expected Returns');
        plt.title('Efficient Frontier');
        plt.colorbar(label='Edge Sharpe Ratio')
        # plt.show();
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
        plot_data = buffer.getvalue()
        imb = base64.encodebytes(plot_data)   
        ims = imb.decode()
        plt.close()
        return ims;