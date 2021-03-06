from django.http import HttpResponse
from django.shortcuts import render
from pandas.core.algorithms import mode
from webroot.invest.markowitz import Markowitz;
import datetime;
import matplotlib
from webroot.invest.excelDatasource import ExcelDatasource;
def index(request):
    matplotlib.use('Agg')
    context = {};
    # selected = ['LI', 'NIO', 'XPEV'];
    selected =["000623.SZ", "002736.SZ", "600030.SH"];
    stardards_code = ["000001.SH", "uso.NYSE"];
    risk_free_national_debt = '^FVX';
    # start_time = datetime.datetime(2020,9,1);
    # end_time = datetime.datetime(2022,7,1);
    # start_time = datetime.datetime(2015,1,5);
    start_time = datetime.datetime(2018,5,30);
    end_time = datetime.datetime(2019,5,30);
    datasource = ExcelDatasource(start_time, end_time, "/Users/linwaiwai/Documents/私人/MBA/IN/sample/",selected, stardards_code);
    markowitz = Markowitz(selected, risk_free_national_debt, start_time, end_time, datasource);
    month_changes = markowitz.getMonthChanges();
    models = markowitz.getModels(month_changes);

    pricePlot = markowitz.getPricePlot();
    pricePlotSrc = 'data:image/png;base64,' + str(pricePlot)

    portfolio = markowitz.getPortfolio();
    portfolioPlot = portfolio['ims'];
    min_variance_port = portfolio["min_variance_port"];
    sharpe_portfolio = portfolio["sharpe_portfolio"];
    portfolioPlotSrc = 'data:image/png;base64,' + str(portfolioPlot)

    opts = markowitz.getSharpBySLSQP();

    t_point =  markowitz.getTangentPoint();
    tartgetPlot = markowitz.getTargetPlot(t_point[2], t_point[3], portfolio['df']);
    tartgetPlotSrc = 'data:image/png;base64,' + str(tartgetPlot)
    # 画股票价格图
    context['pricePlot'] = pricePlotSrc;
    # 投资模型
    context['models'] = models;
    # 投资组合图
    context['portfolioPlot'] = portfolioPlotSrc;
    # 根据投资组合图得出最大夏普和最小风险点
    context['min_variance_port'] = min_variance_port;
    context['sharpe_portfolio'] = sharpe_portfolio;
    # 通过SLSQP算法计算最佳投资比率；
    context["portfolioBySQP"] = opts['x'];
    # 画出有效边界图
    context['tartgetPlotSrc'] = tartgetPlotSrc;
    


    context["t_point"] = t_point;
    return render(request, 'report.html', context)