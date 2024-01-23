import numpy as np

class sobolIndices():
    def __init__(
            self, 
            model: None,
            params: dict=None,
            model_params=None,
            N=10000) -> None:

        """
        Args:
             model (function): function that takes the parameters defined in input_dict and optional params from model_params.
             params (dict): dictionary of inputParam objects for global sensitivity analysis of the form:
                  params = {'param1' : inputParam, ..., 'paramX' : inputParam}
        """

        self.model = model
        self.params = params
        self.model_params = model_params
        self.N = N

        print(self.N)

        self.num_params = np.sum([param.num_params for _,param in self.params.items()])

    def sample_params(self):
        """
        Draw self.N samples from the input parameter priors in self.params 
        """

        for _,param in self.params.items():
            param.sample(self.N)

    def _construct_sample_matrix(self):
        """
        Construct matrix of samples from Saltelli et al.
        """

        # get samples
        self.sample_params()

        samp_matrix = np.zeros([self.N, self.num_params])

        i = 0
        for _,param in self.params.items():
            samp_matrix[:,i:i+param.num_params] = np.reshape(param.samples, [self.N, param.num_params])

            i+=param.num_params

        return samp_matrix

    def construct_A(self):
        """
        Construct A matrix from Saltelli et al.
        """
        
        self.A = self._construct_sample_matrix()

    def construct_B(self):
        """
        Construct B matrix from Saltelli et al.
        """
        
        self.B = self._construct_sample_matrix()

    def construct_A_B(self):
        """
        Construct A matrix with jth column swapped out for the jth column of B
        """

        i = 0
        for key, param in self.params.items():
            param.A_B = np.copy(self.A)
            param.A_B[:,i:i+param.num_params] = self.B[:,i:i+param.num_params]

            i+=param.num_params

    def construct_B_A(self):
        """
        Construct B matrix with jth column swapped out for the jth column of A
        """

        i = 0
        for key, param in self.params.items():
            param.B_A = np.copy(self.B)
            param.B_A[:,i:i+param.num_params] = self.A[:,i:i+param.num_params]

            i+=param.num_params

    def _calculate_f_A(self):
        """
        Calculate f_A using the model
        """
        self.f_A = self._f_total[:self.N]

    def _calculate_f_B(self):
        """
        Calculate f_B using the model
        """
        self.f_B = self._f_total[self.N:2*self.N]

    def _calculate_f_A_B(self):
        """
        Calculate f_A_B for each input parameter using the model
        """
        i=2
        for _,param in self.params.items():
            param.f_A = self._f_total[i*self.N:(i+1)*self.N]
            i+=1

    def _calculate_f_B_A(self):
        """
        Calculate f_B_A for each input parameter using the model
        """
        i = 2 + self.num_params
        for _,param in self.params.items():
            param.f_B = self._f_total[i*self.N:(i+1)*self.N]
            i+=1


    def _construct_AB_total(self):
        """
        Construct matrix of the form [A, B, A_B, B_A]^T
        """

        tmp_AB_total = np.vstack([self.A, self.B])    

        # stack the A_B matrices
        for _,param in self.params.items():
            tmp_AB_total = np.vstack([tmp_AB_total, param.A_B])

        # stack the B_A matrices
        for _,param in self.params.items():
            tmp_AB_total = np.vstack([tmp_AB_total, param.B_A])

        self._AB_total = tmp_AB_total

    def _calculate_f_total(self):
        """
        Calculate f for each A, B, A_B, and B_A matrix
        """

        self._construct_AB_total()

        if self.model_params is not None:
            self._f_total = self.model(self._AB_total, self.model_params)
        else:
            self._f_total = self.model(self._AB_total)

    def calculate_f(self):
        """
        Calculate f for each A, B, A_B, and B_A matrix
        """

        self._calculate_f_total()

        self._calculate_f_A()
        self._calculate_f_B()
        self._calculate_f_A_B()
        self._calculate_f_B_A()

    def first_order_indices(self):
        """
        Calculate the first order sobol indices
        """

        for _,param in self.params.items():
            numerator = 1/self.N * np.sum(self.f_B * (param.f_A - self.f_A))
            param.s1 = numerator/self.varX

    def total_indices(self):
        """
        Calculate the total sobol indices
        """

        for _,param in self.params.items():
            numerator = 1/(2*self.N) * np.sum((self.f_A - param.f_A)**2)
            param.sT = numerator/self.varX



    def total_variance(self):
        """
        Calculate Var(X)
        """

        self.varX = 1/(2*self.N) * np.sum((self.f_A - np.mean(self.f_A))**2 + (self.f_B - np.mean(self.f_B))**2)

    
    def sobol_calc(self):
        """
        Calculate the first order and total sobol indices
        """

        # construct the necessary matrices
        self.construct_A()
        self.construct_B()
        self.construct_A_B()
        self.construct_B_A()

        # calculate the f vectors
        self.calculate_f()

        # calculate the total variance
        self.total_variance()

        # calculate the first order indices
        self.first_order_indices()

        # calculate the total indices
        self.total_indices()
