# SVM_Python 
#### Using SMO algorithm

Author: Arjun Kashyap
<br>
January 2020

## Usage
`options = {
    "kernel": "rbf",
    "rbf_sigma": 0.5
}`
<br>
`data = [[0,0], [0,1], [1,0], [1,1]]`
<br>
`labels = [-1, 1, 1, -1]`
<br>
`svm = SVM()`
<br>
`svm.train(data, labels, options)`

For training with different parameters:
Quoting [@karpathy](https://github.com/karpathy)
<br>
"
Rules of thumb: You almost always want to try the linear SVM first and see how that works. You want to play around with different values of C from about 1e-2 to 1e5, as every dataset is different. C=1 is usually a fairly reasonable value. Roughly, C is the cost to the SVM when it mis-classifies one of your training examples. If you increase it, the SVM will try very hard to fit all your data, which may be good if you strongly trust your data. In practice, you usually don't want it too high though. If linear kernel doesn't work very well, try the rbf kernel. You will have to try different values of both C and just as crucially the sigma for the gaussian kernel.

The linear SVM should be much faster than SVM with any other kernel. If you want it even faster but less accurate, you want to play around with options.tol (try increase a bit). You can also try to decrease options.maxiter and especially options.numpasses (decrease a bit). If you use non-linear svm, you can also speed up the svm at test by playing around with options.alphatol (try increase a bit).
"

# Credits
[Fast training support vector classifiers](https://papers.nips.cc/paper/1855-fast-training-of-support-vector-classifiers.pdf)
<br>
[Simplified SMO](http://math.unt.edu/~hsp0009/smo.pdf)
<br>
This repo is the Python implementation of to Andrej Karpathy's [repo](https://github.com/karpathy/svmjs)

# Licence
MIT