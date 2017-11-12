import pandas as pd
import numpy as np

                  
class BinaryBarrecaTransform():
	"""
	Implementation of Daniele Micci-Barreca transformation of High-Cardinality
	Categorical Data for Binary Classification.
	"""

	def __init__(self, min_sample_size, transition_rate):
		"""
		Parameters:
			min_sample_size: minimal sample size in a given category for which we can trust
							 the posterior probability estimate.
			
			transition_rate: controls the rate of transition between the full trust in the posterior
			probability and the prior probability.
	
		"""
		self.min_sample_size = min_sample_size
		self.transition_rate = transition_rate

	
	def _calc(self, X, y):
		"""
		Return:
			Dict in the form: {'prior': probability, 'category1': probability, 'category2': probability, ..., 
			'categoryN': probability}
		"""
		# computing prior
		prior = y.mean()
		
		tr_df = pd.concat([X, y], axis=1)
		
		grp = tr_df.groupby(by=X.name)[y.name].agg(["mean", "count"]).reset_index()
		smoothing = 1 / (1 + np.exp(-(grp["count"] - self.min_sample_size) / self.transition_rate))
		grp[X.name+'_mean'] = prior * (1 - smoothing) + grp["mean"] * smoothing
		grp.drop(["mean", "count"], axis=1, inplace=True)
		return grp

	def fit(self, X, y, **kwargs):
		"""
		Parameters:
			X: Pandas DataFrame with discrete columns.
			y: Binary target as a Pandas Series.
		"""
		assert isinstance(X, pd.core.frame.DataFrame), "X must be a pandas DataFrame"
		assert isinstance(y, pd.core.series.Series), "y must be a pandas Series."
		assert X.shape[0] == y.shape[0], "X and y must have the same length."
		
		self.columns = kwargs.get("columns", X.columns)
		self.probs = []
		for i in self.columns:
			self.probs.append({i: self._calc(X[i], y)})
		return self	
	
	def transform(self, X):
		"""
		Parameters:
			X: Pandas DataFrame with discrete columns to transform.	
		"""
		assert isinstance(X, pd.core.frame.DataFrame), "X must be a pandas DataFrame"
		#assert np.all(np.equal(X.columns, self.columns)), "X columns and X columns must be the same"
		k = 0
		X = X.copy()
		for i in self.columns:
			X = X.merge(self.probs[k].get(i), on=i, how='left')  		
			k += 1
		return X

