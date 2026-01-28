"""
Quantile Strategy Backtesting Framework

Core backtesting engine for quantile-based long-short equity strategies.
Handles position generation, portfolio accounting, return calculation, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


class QuantileStrategy:
    """
    Flexible quantile-based long-short strategy backtester
    
    Parameters
    ----------
    data : pd.DataFrame
        Daily data with columns: ticker, date, adj_close, and ratio columns
    ratio_name : str
        Name of ratio column to use for ranking
    direction : str, default 'long_high'
        'long_high': long high values, short low values
        'long_low': long low values, short high values
    rebalance_freq : str, default 'M'
        'M' for monthly, 'W' for weekly rebalancing
    top_quantile : float, default 0.9
        Percentile threshold for long positions (0.9 = top 10%)
    bottom_quantile : float, default 0.1
        Percentile threshold for short positions (0.1 = bottom 10%)
    use_changes : bool, default False
        If True, rank on period-over-period changes instead of levels
    position_sizing : str, default 'equal'
        'equal': equal-weighted positions
        'vigintile_double': double weight on most attractive 5%
        'vigintile_half': half weight on least attractive 5%
    funding_rate : float, default 0.05
        Annual funding rate for cash (5%)
    repo_spread : float, default 0.01
        Spread for short rebate (100bp below funding rate)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        ratio_name: str,
        direction: str = 'long_high',
        rebalance_freq: str = 'M',
        top_quantile: float = 0.9,
        bottom_quantile: float = 0.1,
        use_changes: bool = False,
        position_sizing: str = 'equal',
        funding_rate: float = 0.05,
        repo_spread: float = 0.01
    ):
        self.data = data.copy()
        self.ratio_name = ratio_name
        self.direction = direction
        self.rebalance_freq = rebalance_freq
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.use_changes = use_changes
        self.position_sizing = position_sizing
        self.funding_rate = funding_rate
        self.repo_spread = repo_spread
        
        # Validate inputs
        self._validate_inputs()
        
        # Results storage
        self.prepared_data = None
        self.rebalance_dates = None
        self.positions = None
        self.portfolio_returns = None
        self.portfolio_values = None
        self.performance_metrics = None
        self.initial_capital = None
        
    def _validate_inputs(self):
        """Validate input parameters"""
        if self.direction not in ['long_high', 'long_low']:
            raise ValueError("direction must be 'long_high' or 'long_low'")
        
        if self.rebalance_freq not in ['M', 'W']:
            raise ValueError("rebalance_freq must be 'M' (monthly) or 'W' (weekly)")
        
        if not 0 < self.top_quantile <= 1:
            raise ValueError("top_quantile must be between 0 and 1")
        
        if not 0 <= self.bottom_quantile < 1:
            raise ValueError("bottom_quantile must be between 0 and 1")
        
        if self.bottom_quantile >= self.top_quantile:
            raise ValueError("bottom_quantile must be less than top_quantile")
        
        if self.position_sizing not in ['equal', 'vigintile_double', 'vigintile_half']:
            raise ValueError("position_sizing must be 'equal', 'vigintile_double', or 'vigintile_half'")
        
        required_cols = ['ticker', 'date', 'adj_close', self.ratio_name]
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for backtesting
        
        Returns
        -------
        pd.DataFrame
            Prepared data with scoring column
        """
        df = self.data[['ticker', 'date', 'adj_close', self.ratio_name]].copy()
        df = df.dropna(subset=[self.ratio_name])
        df = df.sort_values(['ticker', 'date'])
        
        if self.use_changes:
            # Calculate period-over-period percentage changes
            df[f'{self.ratio_name}_change'] = df.groupby('ticker')[self.ratio_name].pct_change()
            self.scoring_col = f'{self.ratio_name}_change'
            df = df.dropna(subset=[self.scoring_col])
        else:
            self.scoring_col = self.ratio_name
        
        self.prepared_data = df
        return df
    
    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Get rebalancing dates based on frequency
        
        Returns
        -------
        List[pd.Timestamp]
            Sorted list of rebalancing dates
        """
        df = self.prepared_data
        
        if self.rebalance_freq == 'M':
            # Last trading day of each month
            dates = df.groupby(df['date'].dt.to_period('M'))['date'].max()
        else:  # 'W'
            # Last trading day of each week
            dates = df.groupby(df['date'].dt.to_period('W'))['date'].max()
        
        self.rebalance_dates = sorted(dates.values)
        return self.rebalance_dates
    
    def rank_stocks(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Rank stocks and assign positions on a specific date
        
        Parameters
        ----------
        date : pd.Timestamp
            Rebalancing date
            
        Returns
        -------
        pd.DataFrame
            Positions with columns: ticker, date, position, rank, score
        """
        date_data = self.prepared_data[self.prepared_data['date'] == date].copy()
        
        if len(date_data) == 0:
            return pd.DataFrame()
        
        # Rank stocks (0 to 1 scale)
        date_data['rank'] = date_data[self.scoring_col].rank(pct=True)
        
        # Assign positions
        date_data['position'] = 0
        
        if self.direction == 'long_high':
            # Long top quantile, short bottom quantile
            date_data.loc[date_data['rank'] >= self.top_quantile, 'position'] = 1
            date_data.loc[date_data['rank'] <= self.bottom_quantile, 'position'] = -1
        else:  # 'long_low'
            # Long bottom quantile, short top quantile
            date_data.loc[date_data['rank'] <= self.bottom_quantile, 'position'] = 1
            date_data.loc[date_data['rank'] >= self.top_quantile, 'position'] = -1
        
        # Return only stocks with positions
        positions = date_data[date_data['position'] != 0].copy()
        positions['score'] = positions[self.scoring_col]
        
        return positions[['ticker', 'date', 'position', 'rank', 'score']]
    
    def _apply_position_sizing(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position sizing scheme based on rank
        
        Parameters
        ----------
        positions : pd.DataFrame
            Positions with rank information
            
        Returns
        -------
        pd.DataFrame
            Positions with weight column added
        """
        if self.position_sizing == 'equal':
            # Equal weight (default)
            long_count = (positions['position'] == 1).sum()
            short_count = (positions['position'] == -1).sum()
            
            if long_count > 0:
                positions.loc[positions['position'] == 1, 'weight'] = 1.0 / long_count
            if short_count > 0:
                positions.loc[positions['position'] == -1, 'weight'] = -1.0 / short_count
        
        elif self.position_sizing == 'vigintile_double':
            # Double the most attractive vigintile (top 5% of positions)
            positions = self._vigintile_sizing(positions, top_multiplier=2.0, bottom_multiplier=1.0)
        
        elif self.position_sizing == 'vigintile_half':
            # Halve the least attractive vigintile (potential outliers)
            positions = self._vigintile_sizing(positions, top_multiplier=1.0, bottom_multiplier=0.5)
        
        return positions
    
    def _vigintile_sizing(self, positions: pd.DataFrame, top_multiplier: float, bottom_multiplier: float) -> pd.DataFrame:
        """
        Size positions by vigintiles (5% buckets within decile)
        
        Parameters
        ----------
        positions : pd.DataFrame
            Positions with rank
        top_multiplier : float
            Weight multiplier for most attractive 5%
        bottom_multiplier : float
            Weight multiplier for least attractive 5%
            
        Returns
        -------
        pd.DataFrame
            Positions with adjusted weights
        """
        # Long positions
        longs = positions[positions['position'] == 1].copy()
        if len(longs) > 0:
            longs = longs.sort_values('rank', ascending=False)
            
            # Top 5% of longs (most attractive = highest rank)
            top_5pct = max(int(len(longs) * 0.05), 1)
            longs['multiplier'] = bottom_multiplier
            longs.iloc[:top_5pct, longs.columns.get_loc('multiplier')] = top_multiplier
            
            # Normalize weights to sum to 1.0
            total_weight = longs['multiplier'].sum()
            longs['weight'] = longs['multiplier'] / total_weight
        
        # Short positions
        shorts = positions[positions['position'] == -1].copy()
        if len(shorts) > 0:
            shorts = shorts.sort_values('rank', ascending=True)
            
            # Top 5% of shorts (most attractive shorts = lowest rank)
            top_5pct = max(int(len(shorts) * 0.05), 1)
            shorts['multiplier'] = bottom_multiplier
            shorts.iloc[:top_5pct, shorts.columns.get_loc('multiplier')] = top_multiplier
            
            # Normalize weights to sum to -1.0
            total_weight = shorts['multiplier'].sum()
            shorts['weight'] = -shorts['multiplier'] / total_weight
        
        # Combine
        result = pd.concat([longs, shorts], ignore_index=True)
        return result.drop(columns=['multiplier'], errors='ignore')
    
    def generate_positions(self) -> pd.DataFrame:
        """
        Generate all positions across rebalancing dates
        
        Returns
        -------
        pd.DataFrame
            All positions with weighting applied
        """
        all_positions = []
        
        for rebal_date in self.rebalance_dates:
            positions = self.rank_stocks(rebal_date)
            
            if len(positions) > 0:
                # Apply position sizing
                positions = self._apply_position_sizing(positions)
                positions['rebalance_date'] = rebal_date
                all_positions.append(positions)
        
        self.positions = pd.concat(all_positions, ignore_index=True)
        return self.positions
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily portfolio returns
        
        Returns
        -------
        pd.DataFrame
            Daily portfolio returns with columns: date, portfolio_return, long_return, short_return
        """
        # Get all prices
        prices = self.prepared_data[['ticker', 'date', 'adj_close']].copy()
        
        # Merge positions with prices
        # Positions hold from rebalance_date until next rebalance_date
        holdings = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            # Get next rebalance date (or end of data)
            if i < len(self.rebalance_dates) - 1:
                next_rebal = self.rebalance_dates[i + 1]
            else:
                next_rebal = self.prepared_data['date'].max() + pd.Timedelta(days=1)
            
            # Get positions for this period
            period_positions = self.positions[self.positions['rebalance_date'] == rebal_date].copy()
            
            # Get prices for holding period
            period_prices = prices[
                (prices['date'] > rebal_date) & 
                (prices['date'] < next_rebal) &
                (prices['ticker'].isin(period_positions['ticker']))
            ].copy()
            
            # Merge positions with prices
            period_holdings = period_prices.merge(
                period_positions[['ticker', 'weight', 'position']],
                on='ticker',
                how='inner'
            )
            
            holdings.append(period_holdings)
        
        all_holdings = pd.concat(holdings, ignore_index=True)
        all_holdings = all_holdings.sort_values(['ticker', 'date'])
        
        # Calculate daily returns for each position
        all_holdings['daily_return'] = all_holdings.groupby('ticker')['adj_close'].pct_change()
        all_holdings['weighted_return'] = all_holdings['weight'] * all_holdings['daily_return']
        
        # Aggregate to portfolio level
        portfolio_returns = all_holdings.groupby('date').agg({
            'weighted_return': 'sum'
        }).reset_index()
        portfolio_returns.columns = ['date', 'portfolio_return']
        
        # Calculate long and short components separately
        long_returns = all_holdings[all_holdings['position'] == 1].groupby('date')['weighted_return'].sum()
        short_returns = all_holdings[all_holdings['position'] == -1].groupby('date')['weighted_return'].sum()
        
        portfolio_returns = portfolio_returns.merge(
            long_returns.rename('long_return'),
            left_on='date',
            right_index=True,
            how='left'
        ).merge(
            short_returns.rename('short_return'),
            left_on='date',
            right_index=True,
            how='left'
        )
        
        portfolio_returns = portfolio_returns.fillna(0)
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns
    
    def calculate_portfolio_values(self, initial_capital: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate cumulative portfolio values
        
        Parameters
        ----------
        initial_capital : float, optional
            Initial capital. If None, calculated as 10x first period gross notional
            
        Returns
        -------
        pd.DataFrame
            Portfolio values over time
        """
        if self.portfolio_returns is None:
            raise ValueError("Must call calculate_returns() first")
        
        # Calculate initial capital if not provided
        if initial_capital is None:
            # 10x gross notional of first month
            first_rebal = self.rebalance_dates[0]
            first_positions = self.positions[self.positions['rebalance_date'] == first_rebal]
            gross_notional = first_positions['weight'].abs().sum()
            initial_capital = 10 * gross_notional
        
        portfolio_values = self.portfolio_returns.copy()
        portfolio_values['portfolio_value'] = initial_capital * (1 + portfolio_values['portfolio_return']).cumprod()
        portfolio_values['cumulative_return'] = portfolio_values['portfolio_value'] / initial_capital - 1
        
        self.portfolio_values = portfolio_values
        self.initial_capital = initial_capital
        
        return portfolio_values
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio"""
        if self.portfolio_returns is None:
            return np.nan
        
        excess_returns = self.portfolio_returns['portfolio_return'] - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    
    def get_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown
        
        Returns
        -------
        Tuple[float, pd.Timestamp, pd.Timestamp]
            (max_drawdown, start_date, end_date)
        """
        if self.portfolio_values is None:
            return np.nan, None, None
        
        cumulative = self.portfolio_values['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_end_idx = drawdown.idxmin()
        
        # Find start of drawdown
        max_dd_start_idx = cumulative[:max_dd_end_idx].idxmax()
        
        start_date = self.portfolio_values.loc[max_dd_start_idx, 'date']
        end_date = self.portfolio_values.loc[max_dd_end_idx, 'date']
        
        return max_dd, start_date, end_date
    
    def get_downside_deviation(self, mar: float = 0.0) -> float:
        """Calculate downside deviation (semi-deviation below MAR)"""
        if self.portfolio_returns is None:
            return np.nan
        
        returns = self.portfolio_returns['portfolio_return']
        downside_returns = returns[returns < mar] - mar
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_dev = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        return downside_dev
    
    def get_sortino_ratio(self, mar: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if self.portfolio_returns is None:
            return np.nan
        
        excess_return = self.portfolio_returns['portfolio_return'].mean() * 252 - mar
        downside_dev = self.get_downside_deviation(mar)
        
        if downside_dev == 0:
            return 0.0
        
        return excess_return / downside_dev
    
    def get_var_cvar(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR
        
        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
            
        Returns
        -------
        Tuple[float, float]
            (VaR, CVaR) at given confidence level
        """
        if self.portfolio_returns is None:
            return np.nan, np.nan
        
        returns = self.portfolio_returns['portfolio_return']
        var = returns.quantile(1 - confidence)
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def get_downside_beta(self, market_returns: pd.DataFrame, threshold: float = 0.0) -> float:
        """
        Calculate downside beta (beta during market down days)
        
        Parameters
        ----------
        market_returns : pd.DataFrame
            DataFrame with 'date' and 'return' columns for market benchmark
        threshold : float
            Threshold for "down" days (default: 0)
            
        Returns
        -------
        float
            Downside beta coefficient
        """
        if self.portfolio_returns is None:
            return np.nan
        
        # Merge portfolio and market returns
        combined = self.portfolio_returns[['date', 'portfolio_return']].merge(
            market_returns[['date', 'return']],
            on='date',
            how='inner'
        )
        
        if len(combined) == 0:
            return np.nan
        
        # Filter to down days
        down_days = combined[combined['return'] < threshold]
        
        if len(down_days) < 2:
            return np.nan
        
        # Calculate beta
        covariance = down_days['portfolio_return'].cov(down_days['return'])
        market_variance = down_days['return'].var()
        
        if market_variance == 0:
            return np.nan
        
        return covariance / market_variance
    
    def get_pl_to_notional(self) -> Dict:
        """
        Calculate PL relative to traded notional
        
        Returns
        -------
        Dict
            Dictionary with total_pl, traded_notional, and pl_per_dollar_traded
        """
        if self.portfolio_values is None or self.positions is None:
            return {
                'total_pl': np.nan,
                'traded_notional': np.nan,
                'pl_per_dollar_traded': np.nan
            }
        
        # Total PL
        total_pl = self.portfolio_values['portfolio_value'].iloc[-1] - self.initial_capital
        
        # Calculate traded notional (sum of absolute position changes at each rebalance)
        traded_notional = 0
        
        for i in range(len(self.rebalance_dates)):
            current_rebal = self.rebalance_dates[i]
            current_pos = self.positions[self.positions['rebalance_date'] == current_rebal]
            
            if i == 0:
                # First period: traded notional is gross notional
                turnover = current_pos['weight'].abs().sum()
            else:
                # Subsequent periods: calculate turnover
                prev_rebal = self.rebalance_dates[i - 1]
                prev_pos = self.positions[self.positions['rebalance_date'] == prev_rebal]
                
                # Merge to find changes
                merged = current_pos[['ticker', 'weight']].merge(
                    prev_pos[['ticker', 'weight']], 
                    on='ticker', 
                    how='outer', 
                    suffixes=('_new', '_old')
                ).fillna(0)
                
                # Sum of absolute changes
                turnover = (merged['weight_new'] - merged['weight_old']).abs().sum()
            
            traded_notional += turnover
        
        return {
            'total_pl': total_pl,
            'traded_notional': traded_notional,
            'pl_per_dollar_traded': total_pl / traded_notional if traded_notional > 0 else np.nan,
            'turnover_per_rebalance': traded_notional / len(self.rebalance_dates) if len(self.rebalance_dates) > 0 else np.nan
        }
    
    def get_performance_summary(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns
        -------
        Dict
            Dictionary of performance metrics
        """
        if self.portfolio_returns is None or self.portfolio_values is None:
            raise ValueError("Must run full backtest first")
        
        returns = self.portfolio_returns['portfolio_return']
        
        # Basic stats
        total_return = self.portfolio_values['cumulative_return'].iloc[-1]
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe = self.get_sharpe_ratio()
        sortino = self.get_sortino_ratio()
        max_dd, dd_start, dd_end = self.get_max_drawdown()
        var_95, cvar_95 = self.get_var_cvar(0.95)
        downside_dev = self.get_downside_deviation()
        
        # Win/loss stats
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        win_rate = winning_days / len(returns) if len(returns) > 0 else 0
        
        avg_win = returns[returns > 0].mean() if winning_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_days > 0 else 0
        
        # Position stats
        avg_long_positions = (self.positions['position'] == 1).sum() / len(self.rebalance_dates)
        avg_short_positions = (self.positions['position'] == -1).sum() / len(self.rebalance_dates)
        
        # PL to notional
        pl_notional = self.get_pl_to_notional()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_start': dd_start,
            'max_drawdown_end': dd_end,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_dev,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_long_positions': avg_long_positions,
            'avg_short_positions': avg_short_positions,
            'num_rebalances': len(self.rebalance_dates),
            'total_days': len(returns),
            'total_pl': pl_notional['total_pl'],
            'traded_notional': pl_notional['traded_notional'],
            'pl_per_dollar_traded': pl_notional['pl_per_dollar_traded'],
            'turnover_per_rebalance': pl_notional['turnover_per_rebalance']
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def run_backtest(self, initial_capital: Optional[float] = None, verbose: bool = True) -> 'QuantileStrategy':
        """
        Run complete backtest
        
        Parameters
        ----------
        initial_capital : float, optional
            Initial portfolio capital
        verbose : bool, default True
            Print progress messages
            
        Returns
        -------
        QuantileStrategy
            Self (for method chaining)
        """
        if verbose:
            direction_str = "High→Long, Low→Short" if self.direction == 'long_high' else "Low→Long, High→Short"
            change_str = " (Changes)" if self.use_changes else ""
            sizing_str = f" [{self.position_sizing}]" if self.position_sizing != 'equal' else ""
            print(f"Running: {self.ratio_name}{change_str} | {direction_str} | {self.rebalance_freq}{sizing_str}")
        
        # Step 1: Prepare data
        self.prepare_data()
        if verbose:
            print(f"  Data: {len(self.prepared_data):,} observations")
        
        # Step 2: Get rebalancing dates
        self.get_rebalance_dates()
        if verbose:
            start_date = str(self.rebalance_dates[0])[:10]
            end_date = str(self.rebalance_dates[-1])[:10]
            print(f"  Rebalances: {len(self.rebalance_dates)} ({start_date} to {end_date})")
        
        # Step 3: Generate positions
        self.generate_positions()
        if verbose:
            print(f"  Positions: {len(self.positions):,} total")
        
        # Step 4: Calculate returns
        self.calculate_returns()
        if verbose:
            print(f"  Returns: {len(self.portfolio_returns)} days")
        
        # Step 5: Calculate portfolio values
        self.calculate_portfolio_values(initial_capital)
        
        # Step 6: Calculate metrics
        self.get_performance_summary()
        
        if verbose:
            print(f"  ✓ Complete | Sharpe: {self.performance_metrics['sharpe_ratio']:.2f} | "
                  f"Return: {self.performance_metrics['total_return']:.1%} | "
                  f"PL/Notional: {self.performance_metrics['pl_per_dollar_traded']:.3f}")
        
        return self