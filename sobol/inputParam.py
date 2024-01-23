import numpy as np
import pandas as pd

class inputParam():
    def __init__(
            self,
            prior,
            group: bool=False) -> None:
        """
        Args:
             prior (scipy.stats.rv_[any], array, or dataframe) : prior probability distribution of parameter
        """

        self.prior = prior

        self.group = group
        
        if self.group: 
            if not isinstance(self.prior, pd.DataFrame):
                print("Prior incorrectly defined for grouped parameters.")
                return
            
        # define num_params for the inputParam object
        self.count_params()

    def sample(self, N):
        """
        Input:
              N (int) : number of samples to draw
        
        Returns an array of samples drawn from the prior and saved in self.samples
        """

        if isinstance(self.prior, np.ndarray):
            self.samples = np.random.choice(self.prior, size=N)
        elif isinstance(self.prior, pd.DataFrame):
            self.samples = self.prior.sample(n=N, replace=True)
        else:
            self.samples = self.prior.rvs(size=N)

    def count_params(self):
        """
        Return the number of parameters in the inputParam object
        """

        if not self.group:
            self.num_params = 1
        else:
            self.num_params = self.prior.shape[1]
        
