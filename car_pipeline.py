# Car leasing pipeline module


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierDetector:

    # Learns outlier cutoffs on the TRAIN set only and applies them to any set.
    # Price is the target (y), so price rules are applied via y, not X.

    def __init__(self, low_quantile=0.025, iqr_multiplier=2.5, min_year=2000):
        self.low_quantile = low_quantile
        self.iqr_multiplier = iqr_multiplier
        self.min_year = min_year

    @staticmethod
    def _iqr_upper(s, k):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return q3 + k * (q3 - q1)

    def fit(self, X, y):
        self.low_price_cutoff_ = y[y > 0].quantile(self.low_quantile)
        self.high_price_cutoff_ = self._iqr_upper(y, self.iqr_multiplier)
        self.high_odometer_cutoff_ = self._iqr_upper(X['odometer'], self.iqr_multiplier)
        return self

    def price_mask(self, y):
        return (y > self.low_price_cutoff_) & (y < self.high_price_cutoff_)

    def odometer_mask(self, X):
        return X['odometer'].isna() | (X['odometer'] < self.high_odometer_cutoff_)

    def scope_mask(self, X):
        # Fixed business rules (no learned statistics): no liens, year >= min_year.
        return (X['title_status'] != 'lien') & (X['year'] >= self.min_year)

    def mask(self, X, y=None, filter_price=True, filter_odometer=True):
        m = self.scope_mask(X)
        if filter_odometer:
            m &= self.odometer_mask(X)
        if y is not None:
            m &= self.price_mask(y) if filter_price else (y > 0)
        return m

    def transform(self, X, y=None, **kwargs):
        m = self.mask(X, y, **kwargs)
        return (X.loc[m], y.loc[m]) if y is not None else X.loc[m]

class FeatureEngineer:

    def __init__(self, data):
            self.data = data.copy()

    def create_location_clusters(self, n_clusters):
        imputer = SimpleImputer(strategy='mean')
        self.data[['lat', 'long']] = imputer.fit_transform(self.data[['lat', 'long']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['location_cluster'] = kmeans.fit_predict(self.data[['lat', 'long']])
        return self.data
    
    def create_age_feature(self):
        current_year = 2022
        self.data['age'] = current_year - self.data['year']
        self.data['age'] = self.data['age'].clip(lower=1)
        return self.data

    def create_mileage_per_year_feature(self):
        self.data['mileage_per_year'] = self.data['odometer'] / self.data['age'].clip(lower=1)
        return self.data

    def add_model_frequency(self):
        model_counts = self.data['model'].value_counts()
        self.data['model_frequency'] = self.data['model'].map(model_counts)
        return self.data

    def rare_categories_grouping(self, column, threshold=0.01):
        value_counts = self.data[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        self.data[column] = self.data[column].replace(rare_categories, 'Other')
        return self.data

    @staticmethod
    def get_engineered_data(data):
        return data


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    # sklearn transformer: everything data-dependent (cluster centres, model counts,
    # rare category lists) is learned in fit() on the training data only.
    # Used as the first step of a Pipeline so cross-validation refits it per fold.

    def __init__(self, n_clusters=50, rare_columns=('manufacturer', 'condition', 'cylinders', 'fuel', 'transmission', 'type'),
                 rare_threshold=0.01, reference_year=2022, drop_columns=()):
        self.n_clusters = n_clusters
        self.rare_columns = rare_columns
        self.rare_threshold = rare_threshold
        self.reference_year = reference_year
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        self.latlong_imputer_ = SimpleImputer(strategy='mean').fit(X[['lat', 'long']])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(
            self.latlong_imputer_.transform(X[['lat', 'long']]))
        self.model_counts_ = X['model'].value_counts()
        self.rare_categories_ = {}
        for col in self.rare_columns:
            freq = X[col].value_counts(normalize=True)
            self.rare_categories_[col] = list(freq[freq < self.rare_threshold].index)
        return self

    def transform(self, X):
        X = X.copy()
        X['age'] = (self.reference_year - X['year']).clip(lower=1)
        X['mileage_per_year'] = X['odometer'] / X['age']
        X['location_cluster'] = self.kmeans_.predict(
            self.latlong_imputer_.transform(X[['lat', 'long']])).astype(str)
        X['model_frequency'] = X['model'].map(self.model_counts_).fillna(0)
        for col, rare in self.rare_categories_.items():
            X[col] = X[col].where(~X[col].isin(rare), 'Other')
        # drop_columns allows ablation of raw or engineered columns
        return X.drop(columns=['model', *self.drop_columns], errors='ignore')