import numpy as np
import pandas as pd
import scipy.optimize as spo
import matplotlib.pyplot as plt

class Portfolio():
    def __init__(self, start_val, assets, alloc, start_date, end_date):
        self.start_val = start_val
        self.assets = assets
        self.alloc = alloc
        self.start = start_date
        self.end = end_date
        
        self.df = self.get_data()
        self.norm = self.normalize()
        self.port_val = self.calc_port_val()
        
        self.cum_ret = self.calc_cum_ret()
        self.daily_ret, self.avg_daily_ret, self.std_daily_ret = self.calc_daily_ret()
        self.sharpe = self.calc_sharpe()
        
        
    def get_data(self):
        dates = pd.date_range(self.start, self.end)
        df = pd.DataFrame(index=dates)

        if 'SPY' in self.assets:
            ref='SPY'
        else:
            ref = self.assets[0]

        for s in self.assets:
            temp = pd.read_csv("../../data/"+s+".csv",
                                index_col='Date',
                                parse_dates=True,
                                usecols=['Date','Adj Close'],
                                na_values=['nan'])
            temp = temp.rename(columns={'Adj Close':s})

            if s == ref:
                df = df.join(temp, how='inner') #use ref stock as reference
            else:
                df = df.join(temp) #default how='left'
        return df

    def normalize(self):
        df = self.df/self.df.iloc[0]
        return df

    def plot_data(dfs, title="Stock Data"):
        for df in dfs:
            df.plot(title=title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

    def calc_port_val(self, alloc=None):
        if alloc == None:
            alloc = self.alloc
        values = self.norm*alloc*self.start_val
        port_val = values.sum(axis=1)
        return port_val

    def get_port_val(self, alloc):
        if alloc == None:
            return self.port_val
        else:
            return self.calc_port_val(alloc)
            
    def calc_cum_ret(self, alloc=None):
        port_val = self.get_port_val(alloc)
            
        cum_ret = port_val/port_val[0] - 1
        return cum_ret

    def calc_daily_ret(self, alloc=None):
        port_val = self.get_port_val(alloc)
        daily_ret = port_val/port_val.shift(1) - 1
        daily_ret = daily_ret[1:]
        avg_daily_ret = daily_ret.mean()
        std_daily_ret = daily_ret.std()

        return daily_ret, avg_daily_ret, std_daily_ret

    def calc_sharpe(self, alloc=None): 
        if alloc == None:
             avg_daily_ret, std_daily_ret = self.avg_daily_ret, self.std_daily_ret 
        else:
            _, avg_daily_ret, std_daily_ret = self.calc_daily_ret(alloc)
            
        sharpe = (avg_daily_ret)/std_daily_ret
        return sharpe

    def sharpe_optimize(self):
        def evaluate(X):
            sharpe = self.calc_sharpe(X)
            #cum_ret = calc_cum_ret(X)[-1]
            #return -(sharpe*0.99 + cum_ret*0.01) #alternate optimization metric metric
            return -sharpe #we will be maximizing sharpe, hence minimizing -sharpe
            
        self.best = spo.minimize(evaluate, [1/len(self.alloc)]*len(self.alloc), 
                    method='SLSQP', 
                    bounds=[(0.0,1.0)]*len(self.alloc), 
                    constraints={'type':'eq', 'fun': lambda x: x.sum() - 1})
        
        return Portfolio(self.start_val, self.assets.copy(), self.best.x.copy(), self.start, self.end)
    
    def mvo_optimize(self,target):
        def evaluate(X):
            _,_,std_daily_ret = self.calc_daily_ret(X)
            return std_daily_ret 
        
        def meet_target(X):
            _, avg_daily_ret, _ = self.calc_daily_ret(X)
            return avg_daily_ret - target
        
        self.best = spo.minimize(evaluate, [1/len(self.alloc)]*len(self.alloc), 
                    method='SLSQP', 
                    bounds=[(0.0,1.0)]*len(self.alloc), 
                    constraints=[{'type':'eq', 'fun': lambda x: x.sum() - 1},
                                 {'type':'eq', 'fun': meet_target}])
        
        return Portfolio(self.start_val, self.assets.copy(), self.best.x.copy(), self.start, self.end)

    def compare(oldP, newP, old="Old Allocation", new="New Allocation"):
        assert oldP.assets == newP.assets
        print("Symbols: {}\n{}: {}\n{}: {}".format(oldP.assets, oldP.alloc, old, new, np.round(newP.alloc,2)))

        
        oldP.cum_ret.plot(label=old)
        newP.cum_ret.plot(label=new)
        plt.title('Cumulative Portfolio Returns')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        print("Old Avg Daily Return:",oldP.avg_daily_ret)
        print("Old Return for entire 3 year period:",oldP.cum_ret[-1])
        print("Old Sharpe Ratio: ", oldP.sharpe,"\n")
        
        print("New Avg Daily Return:",newP.avg_daily_ret)
        print("New Return for entire 3 year period:",newP.cum_ret[-1])
        print("New Sharpe Ratio: ",newP.sharpe)
