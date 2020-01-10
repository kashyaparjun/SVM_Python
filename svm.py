#########################
# @author Arjun Kashyap
# Version 1.2
#########################

# Libraries used
import numpy as np
import json


class SVM:
    """
    This is an implementation of binary SVM classification using the SMO algorithm.
    Read the README.md file for more information
    """

    def __init__(self):
        self.kernel_results = None
        self.w = None
        self.rbf_sigma = None
        self.kernel_type = None
        self.kernel = None
        self.labels = None
        self.alpha = None
        self.data = None
        self.usew_ = None
        self.D = None
        self.N = None
        self.b = None

    def zeros(self, n):
        """
        Create vector of zeros of length n
        :param n:
        :return: A numpy array of zeros
        """
        return np.zeros(n)

    def randi(self, a, b):
        """
        Generate random integer between a and b (b excluded)
        :param a:
        :param b:
        :return: A random integer
        """
        return np.random.randint(low=a, high=b - 1, dtype=int)

    def randf(self, a, b):
        """
        Generate random floating point number between a and b
        :param a:
        :param b:
        :return: A random float number
        """
        return np.random.uniform(a, b)

    def linear_kernel(self, v1, v2):
        """
        Kernel decision function for values v1 and v2
        :param v1:
        :param v2:
        :return: result of linear kernel function
        """
        s = 0
        for i in range(len(v1)):
            s += v1[i] * v2[i]
        return s

    def rbf_kernel(self, v1, v2):
        """
        Rbf kernel decision function
        :param v1:
        :param v2:
        :param sigma:
        :return: results of rbf kernel function
        """
        sigma = self.rbf_sigma
        diff = [v1[i] - v2[i] for i in range(len(v1))]
        summation = np.sum(np.multiply(diff, diff))
        return np.exp(-summation / (2.0 * sigma * sigma))

    def from_json(self, input_json):
        """
        Function to create SVM from stored JSON
        :param input_json:
        :return: takes json input and prepares the model
        """
        input_json = json.loads(input_json)
        self.N = input_json['N']
        self.D = input_json['D']
        self.b = input_json['b']

        self.kernel_type = json['kernel_type']
        if self.kernel_type == 'linear':
            self.w = input_json['w']
            self.usew_ = True
            self.kernel = self.linear_kernel
        elif self.kernel_type == 'rbf':
            self.rbf_sigma = input_json['rbf_sigma']
            self.kernel = self.rbf_kernel
            self.data = input_json['data']
            self.labels = input_json['labels']
            self.alpha = input_json['alpha']
        else:
            print("ERROR: unrecognized kernel type: " + self.kernel_type)

    def to_json(self):
        """
        Generates a json out of the current model
        :return: json format of model
        """
        if self.kernel_type == 'custom':
            print("Can't save custom kernel models")
            return json.dumps({})

        output_json = {'N': self.N, 'D': self.D, 'b': self.b, 'kernel_type': self.kernel_type}

        if self.kernel_type == 'linear':
            output_json['w'] = self.w
        elif self.kernel_type == 'rbf':
            output_json['rbf_sigma'] = self.rbf_sigma
            output_json['data'] = self.data
            output_json['labels'] = self.labels
            output_json['alpha'] = self.alpha

        return json.dumps(output_json)

    def get_weights(self):
        """
        Calculates weights and bias using alpha, labels and data
        :return: weights 'w' and bias 'b'
        """
        w = [None] * self.D
        for j in range(self.D):
            s = 0.0
            for i in range(self.N):
                s += (self.alpha[i] * self.labels[i] * self.data[i][j])
            w[j] = s
        return {'w': w, 'b': self.b}

    def predict(self, data):
        """
        Predict function to predict y values for input X
        :param data:
        :return: classification into 1 and -1
        """
        margs = self.margins(data)
        margs = [1 if x > 0 else -1 for x in margs]
        return margs

    def kernel_result(self, i, j):
        """
        returns the results from kernel for data points i and j
        :param i:
        :param j:
        :return: kernel result
        """
        if self.kernel_results is not None:
            return self.kernel_results[i][j]
        return self.kernel(self.data[i], self.data[j])

    def margins(self, data):
        """
        Uses margin One function to compute margin for the entire data X
        :param data:
        :return: margins
        """
        margins = [self.margin_one(x) for x in data]
        return margins

    def predict_one(self, inst):
        """
        Making a single prediction instead of an Array
        :param inst:
        :return:
        """
        return 1 if self.margin_one(inst) > 0 else -1

    def margin_one(self, inst):
        """
        Calculating margin of a data point. This is the main prediction function.
        :param inst:
        :return: f
        """
        f = self.b
        # If linear kernel is used, weights are calculated and stored. Hence, usew_ would be True
        if self.usew_:
            f += np.sum(np.multiply(inst, self.w))
        # any other kernel function, including RBF
        else:
            for i in range(self.N):
                f += np.multiply(np.multiply(self.alpha[i], self.labels[i]), self.kernel(inst, self.data[i]))
        return f

    def train(self, data, labels, options):
        """
        Computes and classifies data into labels 1 or -1
        :param data:
        :param labels:
        :param options:
        :return: number of iterations
        """
        self.data = data
        self.labels = labels

        # SVM parameters
        options = options if options else {}
        # C value is used to control the regularization of the model. Find more at
        # https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
        # decrease for more regularization
        C = options['C'] if 'C' in options else 1.0
        # Numerical tolerance of a model.
        # For an SVM to be valid, all values should be greater than or equal to 0. And, atleast one value on each side
        # needs to be equal to 0, which will be our support vectors. Since, getting perfect 0 values are unlikely,
        # a tolerance is used to make some room for calculation.
        tol = options['tol'] if 'tol' in options else 1e-4
        # For space and time efficiency, non-support vectors are truncated. Set this to 0 for exact values.
        # Set it up higher for increased efficienct.
        alphatol = options['alphatol'] if 'alphatol' in options else 1e-7
        # Maximum number of iterations during optimization
        maxiter = options['maxiter'] if 'maxiter' in options else 10000
        # The number of iterations of data with no change before halting the training process
        # Needs to be increased for a higher precision
        numpasses = options['numpasses'] if 'numpasses' in options else 10

        kernel = self.linear_kernel
        self.kernel_type = "linear"
        if "kernel" in options:
            if type(options['kernel']) is str:
                if options['kernel'] == "linear":
                    self.kernel_type = "linear"
                    kernel = self.linear_kernel
                elif options['kernel'] == "rbf":
                    rbf_sigma = options['rbf_sigma'] if 'rbf_sigma' in options else 0.5
                    self.rbf_sigma = rbf_sigma
                    self.kernel_type = "rbf"
                    kernel = self.rbf_kernel
            else:
                self.kernel_type = "custom"
                kernel = options['kernel']

        self.kernel = kernel
        N, self.N = len(data), len(data)
        D, self.D = len(data[0]), len(data[0])
        self.alpha = self.zeros(N)
        self.b = 0.0
        self.usew_ = False

        # Caching kernel computations to reduce recomputations when data is huge
        if 'memoize' in options:
            self.kernel_results = [None] * N
            for i in range(N):
                self.kernel_results[i] = [None] * N
                for j in range(N):
                    self.kernel_results[i][j] = kernel(data[i], data[j])

        # The SMO algorithm starts here
        iter = 0
        passes = 0
        while passes < numpasses and iter < maxiter:
            alpha_changed = 0
            for i in range(N):
                Ei = self.margin_one(data[i]) - labels[i]
                if (np.all(labels[i] * Ei < -tol) and np.all(self.alpha[i] < C)) or (
                        np.all(labels[i] * Ei > tol) and np.all(self.alpha[i] > 0)):
                    j = i
                    # Setting alpha_j to a random value !equal to i
                    while j == i:
                        j = self.randi(0, self.N)
                    Ej = self.margin_one(data[j]) - labels[j]

                    # Calculating L(lower) and H(higher) bounds to stay inside square of length and width 'C'
                    ai = self.alpha[i]
                    aj = self.alpha[j]
                    L, H = 0, C
                    if labels[i] == labels[j]:
                        L = np.maximum(0, ai + aj - C)
                        H = np.minimum(C, ai + aj)
                    else:
                        L = np.maximum(0, aj - ai)
                        H = np.minimum(C, C + aj - ai)

                    if np.abs(L - H) < 1e-4:
                        continue

                    eta = 2 * self.kernel_result(i, j) - self.kernel_result(i, i) - self.kernel_result(j, j)
                    if np.all(eta >= 0):
                        continue

                    # Calculating and updating aplha_i and aplha_j
                    newaj = aj - labels[j] * (Ei - Ej) / eta
                    if np.all(newaj > H):
                        newaj = H
                    if np.all(newaj < L):
                        newaj = L
                    if np.all(np.abs(aj - newaj) < 1e-4):
                        continue
                    self.alpha[j] = newaj
                    newai = ai + labels[i] * labels[j] * (aj - newaj)
                    self.alpha[i] = newai

                    # Updating the bias term
                    b1 = self.b - Ei - labels[i] * (newai - ai) * self.kernel_result(i, i) - labels[j] * (
                            newaj - aj) * self.kernel_result(i, j)
                    b2 = self.b - Ei - labels[i] * (newai - ai) * self.kernel_result(i, j) - labels[j] * (
                            newaj - aj) * self.kernel_result(j, j)

                    self.b = 0.5 * (b1 + b2)
                    if 0 < newai < C:
                        self.b = b1
                    if 0 < newaj < C:
                        self.b = b2

                    alpha_changed += 1

            iter += 1
            if alpha_changed == 0:
                passes += 1
            else:
                passes = 0
            print('iteration: '+str(iter)+" alphaChanged: "+str(alpha_changed))

        # If linear kernel is used the weights are calculated and stored to reduce evaluation time.
        if self.kernel_type == "linear":
            self.w = [None] * D
            for j in range(self.D):
                s = 0.0
                for i in range(self.N):
                    s += self.alpha[i] * labels[i] * data[i][j]
                self.w[j] = s
                self.usew_ = True
        else:
            # This is to remove all alpha where it is equal to 0. As alpha[i] = 0 is irrelevant for further training
            # and testing.
            newdata = []
            newlabels = []
            newalpha = []
            for i in range(self.N):
                if self.alpha[i] > alphatol:
                    newdata.append(self.data[i])
                    newlabels.append(self.labels[i])
                    newalpha.append(self.alpha[i])

            self.data = newdata
            self.labels = newlabels
            self.alpha = newalpha
            self.N = len(self.data)

        trainstats = {'iters': iter}
        return trainstats