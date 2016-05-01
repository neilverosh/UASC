# modified from http://deeplearning.net/tutorial/
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

  y_{pred} = argmax_i P(Y=i|x,W,b)

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    
    either n_layers = 1 and the activation gets directly get softmaxed
    or     n_layers = 2 and there is an additional layer and tanh 
                        nonlinearity before the softmax

    """

    def __init__(self, input, n_in, n_out, n_hidden, n_layers):
        # Initialize the parameters of the logistic regression

        rng = numpy.random.RandomState(1234)
        n2 = n_out if n_layers==1 else n_hidden

        self.W = theano.shared(
            value=numpy.array(rng.uniform(low=-numpy.sqrt(6. / (n_in + n2)), 
            high=numpy.sqrt(6. / (n_in + n2)), size=(n_in, n2)),
                dtype=theano.config.floatX),
            name='W', borrow=True)

        # initialize the biases b 
        self.b = theano.shared(
            value=numpy.zeros(
                (n2,),
                dtype=theano.config.floatX),
            name='b', borrow=True)

        self.params = [self.W, self.b]

        if n_layers==2:
            # matrix applied to the hidden units to generate the output
            self.W_out = theano.shared(
                 value=numpy.array(0.0*rng.uniform(low=-numpy.sqrt(6. / (n_hidden + n_out)), 
                high=numpy.sqrt(6. / (n_hidden + n_out)), size=(n_hidden, n_out)),
                    dtype=theano.config.floatX),
                name='W_out', borrow=True)

            # bias applied to the hidden units while generating the output
            self.b_out = theano.shared(
              value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
              ),
              name='b_out',
              borrow=True
            )
            self.params.append(self.W_out)
            self.params.append(self.b_out)

        else:
            if n_layers!=1:
                raise NotImplementedError('n_layers must be 1 or 2')
        
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        if n_layers==1:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(T.tanh(T.dot(input, self.W) + self.b), self.W_out) + self.b_out)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class neural_classifier(object):
    #def __init__(self):

    def plot_decision_boundary(self, plt, X, Y):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.50
        # Generate a grid of points with distance h between them
        xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = numpy.c_[xx.ravel(), yy.ravel()]
        Z = self.get_p_y(numpy.c_[xx.ravel(), yy.ravel()])[0]
        Z = Z[:,0]
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        fig = plt.figure(figsize=[8,8])
        plt.contourf(xx, yy, Z, 1, cmap=plt.cm.get_cmap('coolwarm', 2))
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)

    def shared_dataset(self, data_xy, borrow=True):
	""" Function that loads the dataset into shared variables

	    The reason we store our dataset in shared variables is to allow
	    Theano to copy it into the GPU memory (when code is run on GPU).
	    Since copying data into the GPU is slow, copying a minibatch everytime
	    is needed (the default behaviour if the data is not in a shared
	    variable) would lead to a large decrease in performance.
	"""

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
				   dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
				     borrow=borrow)
	    # When storing data on the GPU it has to be stored as floats
	    # therefore we will store the labels as ``floatX`` as well
	    # (``shared_y`` does exactly that). But during our computations
	    # we need them as ints (we use labels as index, and if they are
	    # floats it doesn't make sense) therefore instead of returning
	    # ``shared_y`` we will have to cast it to int. This little hack
	    # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def load_data(self, dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	test_set_x, test_set_y = self.shared_dataset(test_set)
	valid_set_x, valid_set_y = self.shared_dataset(valid_set)
	train_set_x, train_set_y = self.shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
		(test_set_x, test_set_y)]
	return rval

    def train(self, learning_rate=0.13, n_epochs=100, print_frequency =10,
			       dataset_path=None, X_train=None, Y_train=None,
                               X_test=None, Y_test=None, X_val=None, Y_val=None,
			       batch_size=600, n_in=28*28, 
                               n_out=10, n_hidden=10, n_layers = 1):
	"""
	Demonstrate stochastic gradient descent optimization of a log-linear
	model

	This is demonstrated on MNIST.

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
			      gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: the path of the MNIST dataset file from
		     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

	"""
        #train_set_x, train_set_y = None, None
        #valid_set_x, valid_set_y = None, None 
        #test_set_x, test_set_y = None, None 

        if dataset_path is not None:
       	    datasets = self.load_data(dataset_path)
	    train_set_x, train_set_y = datasets[0]
	    valid_set_x, valid_set_y = datasets[1]
	    test_set_x, test_set_y = datasets[2]
        elif X_train is not None and Y_train is not None:
            train_set_x, train_set_y = self.shared_dataset([X_train, Y_train])
            if X_test is not None and Y_test is not None:
                test_set_x, test_set_y = self.shared_dataset([X_test, Y_test])
            else:
                test_set_x, test_set_y = train_set_x, train_set_y
                print "using training set as test set..." 
            if X_val is not None and Y_val is not None:
                valid_set_x, valid_set_y = self.shared_dataset([X_val, Y_val])
            else:
                valid_set_x, valid_set_y = train_set_x, train_set_y
                print "using training set as validation set..." 
        else:
            raise NotImplementedError()

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        #if valid_set_x is not None:
       	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        #if test_set_x is not None:
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch

	# generate symbolic variables for input (x and y represent a
	# minibatch)
	x = T.matrix('x')  # data, presented as rasterized images
	y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

	# construct the logistic regression class
	# Each MNIST image has size 28*28
	classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out, n_hidden=n_hidden, n_layers=n_layers)

	# the cost we minimize during training is the negative log likelihood of
	# the model in symbolic format
	cost = classifier.negative_log_likelihood(y)

	# compiling a Theano function that computes the mistakes that are made by
	# the model on a minibatch
	test_model = theano.function(
	    inputs=[index],
	    outputs=classifier.errors(y),
	    givens={
		x: test_set_x[index * batch_size: (index + 1) * batch_size],
		y: test_set_y[index * batch_size: (index + 1) * batch_size]
	    }
	)

	self.get_p_y = theano.function(
	    inputs=[x],
	    outputs=[classifier.p_y_given_x])

	self.get_test_pred = theano.function(
	    inputs=[index],
	    outputs=[classifier.y_pred, x, y],
	    givens={
		x: test_set_x[index * batch_size: (index + 1) * batch_size],
		y: test_set_y[index * batch_size: (index + 1) * batch_size]},
	on_unused_input='ignore')

	validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
	    }
	)

	# compute the gradient of cost with respect to theta = (W,b)
        gparams = [T.grad(cost, param) for param in classifier.params]

	# start-snippet-3
	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs.
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams) ]

	# compiling a Theano function `train_model` that returns the cost, but in
	# the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = theano.function(
	    inputs=[index],
	    outputs=cost,
	    updates=updates,
	    givens={
		x: train_set_x[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	    }
	)
	# end-snippet-3

	###############
	# TRAIN MODEL #
	###############
	print '... training the model'
	# early-stopping parameters
	patience = 5000  # look as this many examples regardless
	patience_increase = 4  # wait this much longer when a new best is
				      # found
	improvement_threshold = 0.999  # a relative improvement of this much is
				      # considered significant
	validation_frequency = print_frequency #min(n_train_batches, patience / 2)
				      # go through this many
				      # minibatche before checking the network
				      # on the validation set; in this case we
				      # check every epoch

	best_validation_loss = numpy.inf
	test_score = 0.
	start_time = timeit.default_timer()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs) and (not done_looping):
	    epoch = epoch + 1
	    for minibatch_index in xrange(n_train_batches):

		minibatch_avg_cost = train_model(minibatch_index)
		# iteration number
		iter = (epoch - 1) * n_train_batches + minibatch_index

		if (iter + 1) % validation_frequency == 0:
		    # compute zero-one loss on validation set
		    validation_losses = [validate_model(i)
					 for i in xrange(n_valid_batches)]
		    this_validation_loss = numpy.mean(validation_losses)

        	    print(
			'epoch %i, minibatch %i/%i, validation error %f %%' %
			(
			    epoch,
			    minibatch_index + 1,
			    n_train_batches,
			    this_validation_loss * 100.
			)
                    )

		    # if we got the best validation score until now
		    if this_validation_loss < best_validation_loss:
			#improve patience if loss improvement is good enough
			if this_validation_loss < best_validation_loss *  \
			   improvement_threshold:
			    patience = max(patience, iter * patience_increase)

			best_validation_loss = this_validation_loss
			# test it on the test set

			test_losses = [test_model(i)
				       for i in xrange(n_test_batches)]
			test_score = numpy.mean(test_losses)

                        if (iter+1)% print_frequency ==0:                   
			    print(
			    (
				'epoch %i, minibatch %i/%i, test error of'
				' best model %f %%'
			    ) %
			    (
				epoch,
				minibatch_index + 1,
				n_train_batches,
				test_score * 100.
			    )
                            )

			# save the best model
			#with open('best_model.pkl', 'w') as f:
			#    cPickle.dump(classifier, f)

		if patience <= iter:
		    done_looping = True
		    break

	end_time = timeit.default_timer()
	print(
	    (
		'Optimization complete with best validation score of %f %%,'
		'with test performance %f %%'
	    )
	    % (best_validation_loss * 100., test_score * 100.)
	)
	print 'The code ran for %d epochs, with %f epochs/sec' % (
	    epoch, 1. * epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' +
			      os.path.split(__file__)[1] +
			      ' ran for %.1fs' % ((end_time - start_time)))
