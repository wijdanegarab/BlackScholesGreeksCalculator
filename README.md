# Option Pricing & Greeks Calculator

## Description

Black-Scholes option pricing engine with full Greeks calculation 
for European Call/Put valuation and risk management.

## Exemple
```python
from option import Option


opt = Option(price=150, strike=155, time=0.25, rate=0.05, volatility=0.20, option_type='call')


print(opt.get_price())  # 4.50€


print(opt.get_greeks())  # Delta, Gamma, Vega, Theta, Rho
```

---

## Features

✅ Black-Scholes pricing (CALL & PUT)  
✅ All 5 Greeks (Delta, Gamma, Vega, Theta, Rho)  
✅ Input validation & error handling  
✅ 3 visualizations (Delta, CALL/PUT curves, Time decay)  

---

## Libraries

- NumPy, SciPy, Matplotlib

---

## Results Example

| Metric | Value |
|--------|-------|
| CALL Price | 4.50€ |
| Delta | 0.439 |
| Vega | 10.15 |
| Theta | -0.048 |

---

## Run
```bash
python main.py
```

Print 3 graphs: Delta sensitivity, CALL/PUT comparison, Time decay.
