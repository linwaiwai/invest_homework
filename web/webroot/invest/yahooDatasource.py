from datasource import Datasource;
import pandas_datareader as web
import numpy as np
class YahooDatasource(Datasource):
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
    

    def getDataByCode(self, code):
        data = web.DataReader(code,'yahoo',self.start_time, self.end_time);
        return data;
        
    def getStardard(self):
        code = "^GSPC";
        data = self.getDataByCode(code);
        return data;

    def getMonthDataByCode(self, code):
        # data = web.DataReader(code,'yahoo',start, end)
        data = self.getDataByCode(code);
        adjclose = self.month_change(code, data);
        # plt.title(title);
        # adjclose["Adj Close"].plot();
        # plt.show();
        return adjclose;

    def getMonthStardardData(self):
        data = self.getStardard();
        adjclose = self.month_change("S&P 500", data);
        return adjclose;

    def month_change(self, code, data):
        pct_change = data.resample('1M').mean().pct_change().dropna();
        print("{start_time_str}——{end_time_str}{code}月收益".format(code=code, start_time_str=self.start_time.strftime("%Y年%m月%d日"), end_time_str=self.end_time.strftime("%Y年%m月%d日")));
        print(pct_change);
        adjclose = np.array(pct_change["Adj Close"]);
        # plt.title(title);
        # adjclose["Adj Close"].plot();
        # plt.show();
        return adjclose;

    def getDatas(self):
        data = web.DataReader(self.selected,'yahoo', self.start_time, self.end_time);
        data = data.stack(level = 1).reset_index(level = [0, 1], drop = False).reset_index();
        df = data[['Date','Adj Close','Symbols']];
        data = df.rename(columns={'Date':'date', 'Symbols':'ticker', 'Adj Close':'adj_close'})
        return data;
        