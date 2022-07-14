from markowitz import Markowitz;
import datetime;
from yahooDatasource import YahooDatasource;
from excelDatasource import ExcelDatasource;
import matplotlib.pyplot as plt

# selected = ['LI', 'NIO', 'XPEV'];
# titles = ["小鹏", '蔚来', '理想'];
risk_free_national_debt = '^FVX';
start_time = datetime.datetime(2018,4,30);
end_time = datetime.datetime(2019,4,30);
# start_time = datetime.datetime(2020,9,1);
# end_time = datetime.datetime(2022,7,1);
selected =["000623.SZ", "002736.SZ", "600030.SH"]

# datasource = YahooDatasource(start_time, end_time);
datasource = ExcelDatasource(start_time, end_time, "/Users/linwaiwai/Documents/私人/MBA/IN/sample/",selected, "000001.SH");
# datasource.getData();
adjclose = datasource.getMonthDataByCode("000623.SZ");
# adjclose["Adj Close"].plot();
# plt.show();
markowitz = Markowitz(selected, risk_free_national_debt, start_time, end_time, datasource);
month_changes = markowitz.getMonthChanges();
models = markowitz.getModels(month_changes);
pricePlot = markowitz.getPricePlot();

portfolio = markowitz.getPortfolio();
portfolioPlot = portfolio['ims'];
min_variance_port = portfolio["min_variance_port"];
sharpe_portfolio = portfolio["sharpe_portfolio"];
t_point =  markowitz.getTangentPoint();

tartgetPlot = markowitz.getTargetPlot(t_point[2], t_point[3], portfolio['df']);
if (t_point[0][0] < 0):
    print("无法求出结果");
