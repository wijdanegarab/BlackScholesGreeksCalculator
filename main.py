from option import Option
from monte_carlo import MonteCarlo
import matplotlib.pyplot as plt
import numpy as np


mc = MonteCarlo(150, 155, 0.25, 0.05, 0.20, 'call')
comparison = mc.compare_with_bs(simul=50000)

print(f"BS: ${comparison['bs_price']:.6f} | MC: ${comparison['mc_price']:.6f} | Error: {comparison['relative_error']:.4f}%")


prices = np.linspace(135, 175, 100)
delta_values = [Option(p, 155, 0.25, 0.05, 0.20, 'call').get_greeks()['delta'] for p in prices]
call_prices = [Option(p, 155, 0.25, 0.05, 0.20, 'call').get_price() for p in prices]
put_prices = [Option(p, 155, 0.25, 0.05, 0.20, 'put').get_price() for p in prices]

temps = np.linspace(0.25, 0.01, 100)
call_temps = [Option(150, 155, t, 0.05, 0.20, 'call').get_price() for t in temps]


n_sims_range = [100, 500, 1000, 5000, 10000, 50000]
mc_prices, mc_errors = [], []
for n in n_sims_range:
    paths = mc.generate_paths(simul=n, step=100)
    prix_finaux = paths[:, -1]
    gains = mc.gain(prix_finaux)
    mc_price = np.exp(-mc.r * mc.t) * np.mean(gains)
    mc_error = np.exp(-mc.r * mc.t) * np.std(gains) / np.sqrt(n)
    mc_prices.append(mc_price)
    mc_errors.append(mc_error)


fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot(2, 3, 1)
ax1.plot(prices, delta_values, 'r-', linewidth=2)
ax1.axvline(155, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Stock Price (€)'), ax1.set_ylabel('Delta')
ax1.set_title('Delta Sensitivity'), ax1.grid(alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(prices, call_prices, 'g-', linewidth=2, label='CALL')
ax2.plot(prices, put_prices, 'r-', linewidth=2, label='PUT')
ax2.axvline(155, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Stock Price (€)'), ax2.set_ylabel('Option Value (€)')
ax2.set_title('CALL vs PUT'), ax2.legend(), ax2.grid(alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(temps, call_temps, 'b-', linewidth=2)
ax3.set_xlabel('Time to Expiration (years)'), ax3.set_ylabel('CALL Value (€)')
ax3.set_title('Time Decay'), ax3.grid(alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
ax4.errorbar(n_sims_range, mc_prices, yerr=mc_errors, marker='o', linestyle='-', color='purple', linewidth=2.5, markersize=8, capsize=5)
ax4.axhline(comparison['bs_price'], color='green', linestyle='--', linewidth=2.5, label=f'BS: ${comparison["bs_price"]:.4f}')
ax4.set_xlabel('Simulations'), ax4.set_ylabel('Price (€)')
ax4.set_title('MC Convergence'), ax4.set_xscale('log'), ax4.legend(), ax4.grid(alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
paths = mc.generate_paths(simul=100, step=100)
times = np.linspace(0, 0.25, 101)
for path in paths:
    ax5.plot(times, path, alpha=0.1, color='blue')
ax5.axhline(155, color='red', linestyle='--', linewidth=2, label='Strike')
ax5.set_xlabel('Time (years)'), ax5.set_ylabel('Stock Price (€)')
ax5.set_title('100 Simulated Paths'), ax5.legend(), ax5.grid(alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
ax6.hist(paths[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
ax6.axvline(np.mean(paths[:, -1]), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(paths[:, -1]):.2f}')
ax6.axvline(155, color='red', linestyle='--', linewidth=2, label='Strike')
ax6.set_xlabel('Final Price (€)'), ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Final Prices'), ax6.legend(), ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('option_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
