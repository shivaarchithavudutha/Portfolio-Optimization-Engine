import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class PortfolioOptimizer:
    def __init__(self, tickers, period='2y'):
        self.tickers = tickers
        self.data = yf.download(tickers, period=period, progress=False, auto_adjust=True)['Close']
        self.returns = self.data.pct_change().dropna()

    def run_simulation(self, num_portfolios=10000):
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            p_return = np.sum(mean_returns * weights) * 252
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            
            results[0,i] = p_return
            results[1,i] = p_std
            results[2,i] = p_return / p_std 

        return results, weights_record

# Execution
assets = ['JPM', 'AAPL', 'GOOGL', 'TSLA', 'GLD']
opt = PortfolioOptimizer(assets)
results, weights = opt.run_simulation()

max_sharpe_idx = np.argmax(results[2])
best_w = weights[max_sharpe_idx]

print(f"Simulation Complete for Portfolio: {assets}")
print("-" * 40)
print("OPTIMAL ASSET ALLOCATION (Max Sharpe Ratio)")
print("-" * 40)
for ticker, weight in zip(assets, best_w):
    print(f"{ticker:8}: {weight:.2%}")
print("-" * 40)
print(f"Expected Annual Return: {results[0, max_sharpe_idx]:.2%}")
print(f"Annual Volatility:    {results[1, max_sharpe_idx]:.2%}")
print("-" * 40)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='plasma', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color='red', marker='*', s=200, label='Max Sharpe Portfolio')
plt.title('Modern Portfolio Theory: Efficient Frontier Analysis', fontsize=14)
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
