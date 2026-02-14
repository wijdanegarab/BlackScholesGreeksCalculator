import numpy as np 
from scipy.stats import norm 

class Option:
    def __init__(self, price, strike, time, rate, volatility, option_type):

        if time <=0:
            raise ValueError("Time must be >0")
        if price <=0:
            raise ValueError("Price must be >0")
        if strike <= 0:
            raise ValueError("Strike must be > 0")
        if volatility <= 0:
            raise ValueError("Volatility must be > 0")
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        
        

        self.s = price 
        self.k = strike 
        self.t = time 
        self.r = rate
        self.sigma = volatility
        self.type = option_type

    def calculate_d1_d2(self):
        d1 = (np.log(self.s/self.k) + (self.r + (self.sigma**2)/2)*self.t) / (self.sigma*np.sqrt(self.t))
        d2 = d1 - self.sigma*np.sqrt(self.t)
        self.d1 = d1
        self.d2 = d2
        return d1, d2
    
    def get_price(self):
        self.calculate_d1_d2()  
        if self.type == 'call':
            price = self.s*norm.cdf(self.d1) - self.k*np.exp(-self.r*self.t)*norm.cdf(self.d2)
        else:  # type == 'put'
            price = self.k*np.exp(-self.r*self.t)*norm.cdf(-self.d2) - self.s*norm.cdf(-self.d1)
        
        self.price = price
        return price
    
    def get_greeks(self):
        self.calculate_d1_d2()  
    
        delta = norm.cdf(self.d1)
    
   
        gamma = norm.pdf(self.d1) / (self.s * self.sigma * np.sqrt(self.t))
    
    
        vega = self.s * norm.pdf(self.d1) * np.sqrt(self.t)
    
    
        if self.type == 'call':
            theta = -self.s * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.t)) - self.r * self.k * np.exp(-self.r*self.t) * norm.cdf(self.d2)
        else:  
            theta = -self.s * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.t)) + self.r * self.k * np.exp(-self.r*self.t) * norm.cdf(-self.d2)
    
    
        if self.type == 'call':
            rho = self.k * self.t * np.exp(-self.r*self.t) * norm.cdf(self.d2)
        else:  
            rho = -self.k * self.t * np.exp(-self.r*self.t) * norm.cdf(-self.d2)
    
   
        self.greeks = {'delta': delta,'gamma': gamma,'vega': vega,'theta': theta,'rho': rho}
    
        return self.greeks
    




    
