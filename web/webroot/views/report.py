from django.http import HttpResponse
from django.shortcuts import render
from pandas.core.algorithms import mode
from webroot.invest.markowitz import Markowitz;
import datetime;
def index(request):
    context = {};
    selected = ['LI', 'NIO', 'XPEV'];
    titles = ["小鹏", '蔚来', '理想'];
    risk_free_national_debt = '^FVX';
    start_time = datetime.datetime(2020,9,1);
    end_time = datetime.datetime(2022,7,1);
    markowitz = Markowitz(selected, titles, risk_free_national_debt, start_time, end_time);
    table = markowitz.getData();
    month_changes = markowitz.getMonthChanges();
    models = markowitz.getModels(month_changes);

    pricePlot = markowitz.getPricePlot(table);
    pricePlotSrc = 'data:image/png;base64,' + str(pricePlot)

    portfolioPlot = markowitz.getPortfolioPlot(table);
    portfolioPlotSrc = 'data:image/png;base64,' + str(portfolioPlot)
    
    context['pricePlot'] = pricePlotSrc;
    context['portfolioPlot'] = portfolioPlotSrc;
    context['month_change'] = month_changes;
    context['models'] = models;
    return render(request, 'report.html', context)