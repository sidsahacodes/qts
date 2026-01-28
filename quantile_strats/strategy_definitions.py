"""
Strategy Definitions for Quantile Backtesting

Defines strategy configurations and combined scoring methods.
All strategies are parameterized - modify parameters in analysis notebook, not here.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable


# ============================================================================
# COMBINED SCORE FUNCTIONS
# ============================================================================

def create_combined_score_zscore(
    data: pd.DataFrame,
    ratios: List[str],
    weights: List[float] = None,
    name: str = 'combined_zscore'
) -> pd.DataFrame:
    """
    Create combined score using z-score standardization
    
    Parameters
    ----------
    data : pd.DataFrame
        Daily data with ratio columns
    ratios : List[str]
        List of ratio column names to combine
    weights : List[float], optional
        Weights for each ratio (must sum to 1). If None, equal weights.
    name : str
        Name for the new combined score column
        
    Returns
    -------
    pd.DataFrame
        Data with new combined score column
    """
    df = data.copy()
    
    # Default to equal weights
    if weights is None:
        weights = [1.0 / len(ratios)] * len(ratios)
    
    if len(weights) != len(ratios):
        raise ValueError("weights must have same length as ratios")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    
    # Calculate z-scores for each ratio on each date
    # Group by date to standardize cross-sectionally
    df_with_scores = []
    
    for date in df['date'].unique():
        date_data = df[df['date'] == date].copy()
        
        # Calculate z-scores
        combined = 0
        for ratio, weight in zip(ratios, weights):
            if ratio in date_data.columns:
                ratio_values = date_data[ratio]
                ratio_mean = ratio_values.mean()
                ratio_std = ratio_values.std()
                
                if ratio_std > 0:
                    z_score = (ratio_values - ratio_mean) / ratio_std
                else:
                    z_score = 0
                
                combined += weight * z_score
        
        date_data[name] = combined
        df_with_scores.append(date_data)
    
    return pd.concat(df_with_scores, ignore_index=True)


def create_combined_score_rank(
    data: pd.DataFrame,
    ratios: List[str],
    directions: List[str] = None,
    weights: List[float] = None,
    name: str = 'combined_rank'
) -> pd.DataFrame:
    """
    Create combined score using rank averaging
    
    Parameters
    ----------
    data : pd.DataFrame
        Daily data with ratio columns
    ratios : List[str]
        List of ratio column names to combine
    directions : List[str], optional
        Direction for each ratio: 'high' (higher is better) or 'low' (lower is better)
        If None, assumes all 'high'
    weights : List[float], optional
        Weights for each ratio (must sum to 1). If None, equal weights.
    name : str
        Name for the new combined score column
        
    Returns
    -------
    pd.DataFrame
        Data with new combined score column
    """
    df = data.copy()
    
    # Default directions
    if directions is None:
        directions = ['high'] * len(ratios)
    
    # Default weights
    if weights is None:
        weights = [1.0 / len(ratios)] * len(ratios)
    
    if len(directions) != len(ratios):
        raise ValueError("directions must have same length as ratios")
    
    if len(weights) != len(ratios):
        raise ValueError("weights must have same length as ratios")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    
    # Calculate ranks for each ratio on each date
    df_with_scores = []
    
    for date in df['date'].unique():
        date_data = df[df['date'] == date].copy()
        
        # Calculate weighted rank average
        combined = 0
        for ratio, direction, weight in zip(ratios, directions, weights):
            if ratio in date_data.columns:
                # Rank from 0 to 1
                if direction == 'high':
                    rank = date_data[ratio].rank(pct=True)
                elif direction == 'low':
                    rank = 1 - date_data[ratio].rank(pct=True)
                else:
                    raise ValueError(f"direction must be 'high' or 'low', got {direction}")
                
                combined += weight * rank
        
        date_data[name] = combined
        df_with_scores.append(date_data)
    
    return pd.concat(df_with_scores, ignore_index=True)


def create_quality_value_score(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined quality-value score
    Quality: High ROI, Low Debt
    Value: Low P/E
    
    Parameters
    ----------
    data : pd.DataFrame
        Daily data with roi, debt_to_mktcap, pe_ratio columns
        
    Returns
    -------
    pd.DataFrame
        Data with 'quality_value' column
    """
    return create_combined_score_rank(
        data,
        ratios=['roi', 'debt_to_mktcap', 'pe_ratio'],
        directions=['high', 'low', 'low'],
        weights=[0.5, 0.25, 0.25],
        name='quality_value'
    )


def create_momentum_score(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create momentum score based on ratio changes
    
    Parameters
    ----------
    data : pd.DataFrame
        Daily data with ratio columns
        
    Returns
    -------
    pd.DataFrame
        Data with momentum columns
    """
    df = data.copy()
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate changes for each ratio
    for ratio in ['roi', 'debt_to_mktcap', 'pe_ratio']:
        if ratio in df.columns:
            df[f'{ratio}_change'] = df.groupby('ticker')[ratio].pct_change()
    
    return df


# ============================================================================
# STRATEGY CONFIGURATIONS
# ============================================================================

def get_base_strategy_configs() -> Dict[str, Dict]:
    """
    Get base strategy configurations (single ratios)
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary of strategy configurations
    """
    strategies = {
        # Strategy 1: ROI - Long profitable, short unprofitable
        'roi_high': {
            'ratio_name': 'roi',
            'direction': 'long_high',
            'description': 'Long High ROI (Profitable), Short Low ROI (Unprofitable)'
        },
        
        # Strategy 2: P/E - Long cheap (value), short expensive
        'pe_low': {
            'ratio_name': 'pe_ratio',
            'direction': 'long_low',
            'description': 'Long Low P/E (Value), Short High P/E (Expensive)'
        },
        
        # Strategy 3: Debt/Mkt Cap - Long low debt (quality), short high debt
        'debt_low': {
            'ratio_name': 'debt_to_mktcap',
            'direction': 'long_low',
            'description': 'Long Low Debt (Quality), Short High Debt (Leveraged)'
        },
    }
    
    return strategies


def get_combined_strategy_configs() -> Dict[str, Dict]:
    """
    Get combined strategy configurations
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary of combined strategy configurations
    """
    strategies = {
        # Strategy 4: Quality + Value combined
        'quality_value': {
            'ratio_name': 'quality_value',
            'direction': 'long_high',
            'description': 'Combined: High ROI + Low Debt + Low P/E',
            'prep_function': create_quality_value_score
        },
    }
    
    return strategies


def get_momentum_strategy_configs() -> Dict[str, Dict]:
    """
    Get momentum-based strategy configurations (changes in ratios)
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary of momentum strategy configurations
    """
    strategies = {
        # ROI momentum - improving vs deteriorating
        'roi_momentum': {
            'ratio_name': 'roi',
            'direction': 'long_high',
            'use_changes': True,
            'description': 'Long Improving ROI, Short Deteriorating ROI'
        },
        
        # P/E momentum
        'pe_momentum': {
            'ratio_name': 'pe_ratio',
            'direction': 'long_low',
            'use_changes': True,
            'description': 'Long P/E Compression, Short P/E Expansion'
        },
    }
    
    return strategies


def get_all_strategy_configs() -> Dict[str, Dict]:
    """
    Get all strategy configurations
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary of all strategy configurations
    """
    all_strategies = {}
    
    all_strategies.update(get_base_strategy_configs())
    all_strategies.update(get_combined_strategy_configs())
    all_strategies.update(get_momentum_strategy_configs())
    
    return all_strategies


def get_strategy_variations(
    base_config: Dict,
    frequencies: List[str] = None,
    quantile_pairs: List[tuple] = None
) -> Dict[str, Dict]:
    """
    Generate variations of a strategy across parameters
    
    Parameters
    ----------
    base_config : Dict
        Base strategy configuration
    frequencies : List[str], optional
        Rebalancing frequencies to test. Default: ['M', 'W']
    quantile_pairs : List[tuple], optional
        (top, bottom) quantile pairs to test. Default: [(0.9, 0.1)]
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary of strategy variations
    """
    if frequencies is None:
        frequencies = ['M']
    
    if quantile_pairs is None:
        quantile_pairs = [(0.9, 0.1)]
    
    variations = {}
    
    for freq in frequencies:
        for top_q, bottom_q in quantile_pairs:
            # Create variation name
            freq_name = 'monthly' if freq == 'M' else 'weekly'
            quantile_name = f'top{int(top_q*100)}_bottom{int(bottom_q*100)}'
            
            var_name = f"{base_config.get('name', 'strategy')}_{freq_name}_{quantile_name}"
            
            # Create config
            var_config = base_config.copy()
            var_config['rebalance_freq'] = freq
            var_config['top_quantile'] = top_q
            var_config['bottom_quantile'] = bottom_q
            
            variations[var_name] = var_config
    
    return variations


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data_for_strategies(
    data: pd.DataFrame,
    strategies: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Prepare data with all required combined scores
    
    Parameters
    ----------
    data : pd.DataFrame
        Base daily data
    strategies : Dict[str, Dict]
        Strategy configurations
        
    Returns
    -------
    pd.DataFrame
        Data with all required score columns added
    """
    df = data.copy()
    
    # Track which prep functions we've already run
    processed_functions = set()
    
    for strategy_name, config in strategies.items():
        if 'prep_function' in config:
            prep_func = config['prep_function']
            
            # Only run each prep function once
            if prep_func.__name__ not in processed_functions:
                print(f"  Preparing {prep_func.__name__}...")
                df = prep_func(df)
                processed_functions.add(prep_func.__name__)
    
    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_strategy_summary(strategies: Dict[str, Dict]):
    """
    Print summary of strategies
    
    Parameters
    ----------
    strategies : Dict[str, Dict]
        Strategy configurations
    """
    print("="*80)
    print(f"STRATEGY CONFIGURATIONS ({len(strategies)} total)")
    print("="*80)
    
    for name, config in strategies.items():
        print(f"\n{name}:")
        print(f"  Ratio: {config['ratio_name']}")
        print(f"  Direction: {config['direction']}")
        if 'use_changes' in config and config['use_changes']:
            print(f"  Uses: Changes (momentum)")
        print(f"  Description: {config['description']}")


def validate_strategy_config(config: Dict) -> bool:
    """
    Validate a strategy configuration
    
    Parameters
    ----------
    config : Dict
        Strategy configuration
        
    Returns
    -------
    bool
        True if valid
    """
    required_keys = ['ratio_name', 'direction', 'description']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Strategy config missing required key: {key}")
    
    if config['direction'] not in ['long_high', 'long_low']:
        raise ValueError(f"Invalid direction: {config['direction']}")
    
    return True
