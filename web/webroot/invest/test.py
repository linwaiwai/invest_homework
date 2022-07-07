from markowitz import Markowitz;
import datetime;

selected = ['LI', 'NIO', 'XPEV'];
titles = ["小鹏", '蔚来', '理想'];
risk_free_national_debt = '^FVX';
start_time = datetime.datetime(2020,9,1);
end_time = datetime.datetime(2022,7,1);
markowitz = Markowitz(selected, titles, risk_free_national_debt, start_time, end_time);

t_point =  markowitz.getTangentPoint();

if (t_point[0][0] < 0):
    print("无法求出结果");
