from .datasource import Datasource;
import pandas as pd;
import numpy as np
class ExcelDatasource(Datasource):
    def __init__(self, start_time, end_time, fold_path, codes, stardard_codes):
        self.start_time = start_time
        self.end_time = end_time
        self.codes = codes;
        self.fold_path = fold_path;
        self.stardard_codes = stardard_codes;

    def getCodeFilePath(self, code):
        file_path = self.fold_path + code + ".csv";
        return file_path;
    

    def getDataByCode(self, code):
        file_path = self.getCodeFilePath(code);
        data = pd.read_csv(file_path);
        data.rename( columns={'Unnamed: 0':'date'}, inplace=True )
        data['ticker'] = code;
        df = data;
        df = df[['date','close','ticker']];
        data = df.rename(columns={'close':'adj_close'});
        # pd.DatetimeIndex(pd.to_datetime(data['date']));
        data['date']=pd.to_datetime(data["date"]);
        data = data[(data['date'] >= self.start_time) & (data['date'] <= self.end_time)]
        data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data['date'])))
        # data.set_index("date");
        return data;

    def getMonthStardardDatas(self):
        # adjcloses = np.ndarray([len(self.stardard_codes), 1]);
        i = 0;
        adjcloses = [];
        for code in self.stardard_codes:
            data = self.getDataByCode(code);
            adjclose = self.month_change(code, data);  
            adjcloses.append(adjclose);
        # adjcloses.reshape(len(self.stardard_codes),len(adjclose));
        adjcloses = np.column_stack(adjcloses);
        return adjcloses;
        
    def getMonthDataByCode(self, code):
        # data = web.DataReader(code,'yahoo',start, end)
         
        data = self.getDataByCode(code);
        adjclose = self.month_change(code, data);
        # plt.title(title);
        # adjclose["Adj Close"].plot();
        # plt.show();
        return adjclose;

    def month_change(self, code, data):
        pct_change = data.resample('1M').mean().pct_change().dropna();
        print("{start_time_str}——{end_time_str}{code}月收益".format(code=code, start_time_str=self.start_time.strftime("%Y年%m月%d日"), end_time_str=self.end_time.strftime("%Y年%m月%d日")));
        print(pct_change);
        adjclose = np.array(pct_change["adj_close"]);
        # plt.title(title);
        # adjclose["Adj Close"].plot();
        # plt.show();
        return adjclose;

    def getDatas(self):
        
        i = 0;
        result = pd.DataFrame();
        for code in self.codes:
            file_path = self.getCodeFilePath(code);
            data = pd.read_csv(file_path);
            data.rename( columns={'Unnamed: 0':'date'}, inplace=True )
            data['ticker'] = code;
            result = pd.concat([result, data]);
        df = result;
        df = df[['date','close','ticker']];
        data = df.rename(columns={'close':'adj_close'});
        data['date'] = pd.to_datetime(data["date"]);
        data = data[(data['date'] >= self.start_time) & (data['date'] <= self.end_time)]
        # data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data['date'])))
        return data;

