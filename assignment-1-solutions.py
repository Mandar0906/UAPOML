import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

print(" \nQUESTION 10")

prices = np.array([
    [100, 108, 103, 115, 110, 119, 125, 121, 130, 127, 135, 140],
    [200, 195, 210, 205, 220, 215, 225, 230, 222, 235, 240, 238],
])

simple_returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print(f"(a) Simple returns shape: {simple_returns.shape}")
print(f"    Stock A returns:\n    {np.round(simple_returns[0], 6)}")
print(f"    Stock B returns:\n    {np.round(simple_returns[1], 6)}")

monthly_mean = simple_returns.mean(axis=1)          
monthly_std  = simple_returns.std(axis=1, ddof=1)   

ann_mean = monthly_mean * 12
ann_std  = monthly_std  * np.sqrt(12)

print(f"\n(b) Annualised mean return — Stock A: {ann_mean[0]:.4f}  "
      f"Stock B: {ann_mean[1]:.4f}")
print(f"    Annualised std dev     — Stock A: {ann_std[0]:.4f}  "
      f"Stock B: {ann_std[1]:.4f}")

cov_matrix = np.cov(simple_returns)
print(f"\n(c) Sample covariance matrix (monthly):\n{cov_matrix}")

rho = np.corrcoef(simple_returns)[0, 1]
sigma_A, sigma_B = monthly_std
manual_cov_AB = rho * sigma_A * sigma_B

print(f"\n    Off-diagonal from np.cov:         {cov_matrix[0, 1]:.8f}")
print(f"    ρ · σ_A · σ_B  (manual check):   {manual_cov_AB:.8f}")
print(f"    Match: {np.isclose(cov_matrix[0, 1], manual_cov_AB)}")

print(" \nQUESTION 11")

RF = 0.04

mu = np.array([0.15, 0.08, 0.05])

sigma_vec = np.array([0.25, 0.12, 0.04])
rho_mat   = np.array([[1.0, 0.4, 0.1],
                      [0.4, 1.0, 0.2],
                      [0.1, 0.2, 1.0]])
Sigma = np.outer(sigma_vec, sigma_vec) * rho_mat

w_eq = np.array([1/3, 1/3, 1/3])

Ep_eq   = w_eq @ mu                        
Var_p_eq = w_eq @ Sigma @ w_eq             
sigma_p_eq = np.sqrt(Var_p_eq)

print(f"(a) Equal-weight portfolio:")
print(f"    E[Rp]  = {Ep_eq:.6f}  ({Ep_eq*100:.4f}%)")
print(f"    σ²_p   = {Var_p_eq:.6f}")
print(f"    σ_p    = {sigma_p_eq:.6f}  ({sigma_p_eq*100:.4f}%)")

N_SIM = 10_000
weights_rand = np.random.dirichlet(np.ones(3), size=N_SIM)

Ep_rand = weights_rand @ mu

sigma_p_rand = np.sqrt(np.einsum('ij,jk,ik->i', weights_rand, Sigma, weights_rand))

print(f"\n(b) Random portfolio simulation:")
print(f"    Shape of Ep_rand:    {Ep_rand.shape}")
print(f"    Shape of sigma_rand: {sigma_p_rand.shape}")
print(f"    E[Rp] range: [{Ep_rand.min():.4f}, {Ep_rand.max():.4f}]")
print(f"    σ_p   range: [{sigma_p_rand.min():.4f}, {sigma_p_rand.max():.4f}]")

sharpe_rand = (Ep_rand - RF) / sigma_p_rand

idx_max_sharpe = np.argmax(sharpe_rand)
max_sharpe     = sharpe_rand[idx_max_sharpe]
best_weights   = weights_rand[idx_max_sharpe]

print(f"\n(c) Sharpe Ratio statistics:")
print(f"    Maximum Sharpe Ratio : {max_sharpe:.4f}")
print(f"    Corresponding weights: "
      f"w1={best_weights[0]:.4f}  w2={best_weights[1]:.4f}  w3={best_weights[2]:.4f}")
print(f"    (Weights sum to: {best_weights.sum():.6f})")

print(" \nQUESTION 12")

mu1, sigma1, w1 = 0.12, 0.20, 0.6
mu2, sigma2     = 0.06, 0.10
w2 = 1 - w1

rho_grid = np.linspace(-1, 1, 200)   # shape (200,)

var_p_grid   = (w1**2 * sigma1**2
              + w2**2 * sigma2**2
              + 2 * w1 * w2 * rho_grid * sigma1 * sigma2)
sigma_p_grid = np.sqrt(var_p_grid)   # shape (200,)

print(f"(a) σ_p array shape: {sigma_p_grid.shape}")
print(f"    σ_p at ρ=-1 : {sigma_p_grid[0]:.6f}")
print(f"    σ_p at ρ=0  : {sigma_p_grid[99]:.6f}")
print(f"    σ_p at ρ=+1 : {sigma_p_grid[-1]:.6f}")

idx_min  = np.argmin(sigma_p_grid)
rho_min  = rho_grid[idx_min]
min_sigp = sigma_p_grid[idx_min]

print(f"\n(b) Minimum σ_p = {min_sigp:.6f}  occurs at ρ = {rho_min:.4f}")

# ------------------------------------------------------------------
# (c) Analytical verification: ∂σ²_p/∂ρ is constant and positive
#
#   σ²_p(ρ) = w1²σ1² + w2²σ2² + 2·w1·w2·σ1·σ2 · ρ
#
#   d(σ²_p)/dρ = 2·w1·w2·σ1·σ2
#
#   With our values: 2 × 0.6 × 0.4 × 0.20 × 0.10 = 0.0096 > 0
#
#   Since the derivative is strictly positive for all ρ, σ²_p (and
#   hence σ_p) is strictly increasing in ρ over [-1, +1].
#   Therefore the minimum is always attained at the LEFT boundary ρ = -1.
# ------------------------------------------------------------------
d_var_d_rho = 2 * w1 * w2 * sigma1 * sigma2
print(f"\n(c) Analytical check:")
print(f"    d(σ²_p)/dρ = 2·w1·w2·σ1·σ2 = {d_var_d_rho:.4f}  (strictly positive)")
print(f"    ⟹ σ²_p is strictly increasing in ρ.")
print(f"    ⟹ Minimum always at ρ = -1 (left boundary).  ✓")
print(f"    At ρ = -1: σ_p = |w1·σ1 - w2·σ2| = "
      f"|{w1}×{sigma1} - {w2}×{sigma2}| = "
      f"{abs(w1*sigma1 - w2*sigma2):.4f}")

print(" \nQUESTION 13")

dates       = pd.date_range('2023-01-02', periods=52, freq='W-MON')
mu_weekly   = np.array([0.003, 0.002, 0.001, 0.0015])
sig_weekly  = np.array([0.04,  0.03,  0.02,  0.025])
returns_sim = np.random.normal(mu_weekly, sig_weekly, (52, 4))
prices_sim  = 100 * np.cumprod(1 + returns_sim, axis=0)
df          = pd.DataFrame(prices_sim, index=dates,
                           columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

df_returns = df.pct_change().dropna()

print(f"(a) First 3 rows of weekly returns DataFrame:")
print(df_returns.head(3).to_string())
print(f"\n    Shape of returns DataFrame: {df_returns.shape}")

stats = df_returns.describe()
print(f"\n(b) Summary statistics:")
print(stats.to_string())

best_mean_asset = df_returns.mean().idxmax()
best_std_asset  = df_returns.std().idxmax()
print(f"\n    Asset with highest mean return : {best_mean_asset} "
      f"({df_returns.mean()[best_mean_asset]:.6f})")
print(f"    Asset with highest std dev     : {best_std_asset} "
      f"({df_returns.std()[best_std_asset]:.6f})")

RF_ANNUAL  = 0.02
RF_WEEKLY  = RF_ANNUAL / 52

weekly_mean_excess = df_returns.mean() - RF_WEEKLY
weekly_std         = df_returns.std()
sharpe_annual      = (weekly_mean_excess / weekly_std) * np.sqrt(52)

print(f"\n(c) Annualised Sharpe Ratios (Rf = 2% p.a.):")
print(sharpe_annual.to_string())

print(" \nQUESTION 14")

corr_matrix = df_returns.corr()
print("(a) Correlation matrix:")
print(corr_matrix.to_string())

upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
)
min_corr_pair  = upper_tri.stack().idxmin()
min_corr_value = upper_tri.stack().min()
print(f"\n    Lowest correlation pair: {min_corr_pair[0]} – {min_corr_pair[1]} "
      f"(ρ = {min_corr_value:.4f})")

weights = pd.Series(0.25, index=df_returns.columns)
port_returns = df_returns.dot(weights)

print(f"\n(b) Equal-weight portfolio returns (first 5 weeks):")
print(port_returns.head().to_string())
print(f"    Mean weekly portfolio return : {port_returns.mean():.6f}")
print(f"    Std  weekly portfolio return : {port_returns.std():.6f}")

monthly_returns = port_returns.resample('ME').apply(
    lambda x: (1 + x).prod() - 1
)

print(f"\n(c) Monthly portfolio returns:")
print(monthly_returns.to_string())
print(f"\n    Mean  of monthly returns: {monthly_returns.mean():.6f}")
print(f"    Stdev of monthly returns: {monthly_returns.std():.6f}")

print(" \nQUESTION 15")

N_PLOT = 20_000
w_plot = np.random.dirichlet(np.ones(3), size=N_PLOT)

Ep_plot    = w_plot @ mu
sigma_plot = np.sqrt(np.einsum('ij,jk,ik->i', w_plot, Sigma, w_plot))
sharpe_plot = (Ep_plot - RF) / sigma_plot

idx_best   = np.argmax(sharpe_plot)
w_best     = w_plot[idx_best]
Ep_best    = Ep_plot[idx_best]
sig_best   = sigma_plot[idx_best]

asset_sigma = np.array([sigma_vec[0], sigma_vec[1], sigma_vec[2]])
asset_mu    = mu
asset_labels = ['Asset 1\n(μ=15%,σ=25%)',
                'Asset 2\n(μ=8%,σ=12%)',
                'Asset 3\n(μ=5%,σ=4%)']

rho_grid2     = np.linspace(-1, 1, 200)
var_p_grid2   = (w1**2 * sigma1**2
               + w2**2 * sigma2**2
               + 2 * w1 * w2 * rho_grid2 * sigma1 * sigma2)
sigma_p_grid2 = np.sqrt(var_p_grid2)
weighted_avg  = w1 * sigma1 + w2 * sigma2          # 0.16

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Portfolio Theory — Week 1 Visualisations', fontsize=14, fontweight='bold')

ax1 = axes[0]

sc = ax1.scatter(sigma_plot * 100, Ep_plot * 100,
                 c=sharpe_plot, cmap='viridis',
                 s=4, alpha=0.5, linewidths=0, zorder=1)

cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
cbar.set_label('Sharpe Ratio', fontsize=10)

ax1.scatter(sig_best * 100, Ep_best * 100,
            marker='*', s=250, color='gold', edgecolors='black',
            linewidths=0.8, zorder=5, label='Max Sharpe Portfolio')

for i in range(3):
    ax1.scatter(asset_sigma[i] * 100, asset_mu[i] * 100,
                s=120, color='#1a1a2e', edgecolors='white',
                linewidths=0.8, zorder=6)
    ax1.annotate(asset_labels[i],
                 xy=(asset_sigma[i] * 100, asset_mu[i] * 100),
                 xytext=(6, 4), textcoords='offset points',
                 fontsize=8, color='#1a1a2e',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

ax1.xaxis.set_major_formatter(mticker.PercentFormatter())
ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
ax1.set_xlabel('Portfolio Risk $\\sigma_p$ (%)', fontsize=10)
ax1.set_ylabel('Expected Return $E[R_p]$ (%)', fontsize=10)
ax1.set_title('Efficient Frontier (20 000 Random Portfolios)', fontsize=11)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.4)

ax2 = axes[1]

ax2.plot(rho_grid2, sigma_p_grid2 * 100, color='steelblue',
         linewidth=2, label='$\\sigma_p(\\rho)$', zorder=3)

ax2.axhline(weighted_avg * 100, color='firebrick', linestyle='--',
            linewidth=1.5, label='Weighted Avg. Risk', zorder=2)

ax2.fill_between(rho_grid2, sigma_p_grid2 * 100, weighted_avg * 100,
                 where=(sigma_p_grid2 * 100 < weighted_avg * 100),
                 color='lightgreen', alpha=0.45,
                 label='Diversification Benefit', zorder=1)

ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_xlabel('Correlation $\\rho$', fontsize=10)
ax2.set_ylabel('Portfolio Risk $\\sigma_p$ (%)', fontsize=10)
ax2.set_title('Portfolio Risk vs. Correlation\n'
              '($w_1=0.6,\\,\\sigma_1=20\\%,\\,w_2=0.4,\\,\\sigma_2=10\\%$)',
              fontsize=11)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(-1, 1)

plt.tight_layout()
plt.savefig('week1_plots.png', dpi=150, bbox_inches='tight')
print("Figure saved as 'week1_plots.png'.")
plt.show()