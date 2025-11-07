"""
CSP-LDA Pipeline
================
Creates a sklearn Pipeline for EEG classification using CSP feature extraction
followed by LDA classification.

Pipeline: CSP → LDA
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP  # type: ignore


def make_pipeline(cfg: Dict[str, Any]) -> Pipeline:
    """
    Create a CSP-LDA pipeline for EEG classification.
    
    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary with hyperparameters:
        - csp_n_components : int, default=6
            Number of CSP components to extract
        - csp_reg : str or float, default='ledoit_wolf'
            Covariance regularization method
        - csp_log : bool, default=True
            Apply log transform to CSP features (log-variance)
        - csp_norm_trace : bool, default=False
            Normalize trace of covariance matrices
        - lda_solver : str, default='lsqr'
            Solver for LDA ('svd', 'lsqr', 'eigen')
        - lda_shrinkage : str or float, default='auto'
            Shrinkage parameter for regularized LDA
    
    Returns
    -------
    Pipeline
        sklearn Pipeline with CSP and LDA steps
    
    Example
    -------
    >>> cfg = {'csp_n_components': 6, 'lda_solver': 'lsqr'}
    >>> pipeline = make_pipeline(cfg)
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_val)
    """
    # Extract CSP hyperparameters with defaults
    csp_n_components = cfg.get("csp_n_components", 6)
    csp_reg = cfg.get("csp_reg", 'ledoit_wolf')
    csp_log = cfg.get("csp_log", True)
    csp_norm_trace = cfg.get("csp_norm_trace", False)
    
    # Extract LDA hyperparameters with defaults
    lda_solver = cfg.get("lda_solver", "lsqr")
    lda_shrinkage = cfg.get("lda_shrinkage", "auto")
    
    # Create CSP transformer
    csp = CSP(
        n_components=csp_n_components,
        reg=csp_reg,
        log=csp_log,                # keep True to output log-variance
        norm_trace=csp_norm_trace,
        transform_into='average_power'  # key change: output average power
    )
    
    # Create LDA classifier
    lda = LDA(
        solver=lda_solver,
        shrinkage=lda_shrinkage
    )
    
    # Create and return pipeline
    return Pipeline([("csp", csp), ("lda", lda)])

