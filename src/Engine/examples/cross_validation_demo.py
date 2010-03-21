import datasets
import error_measures
import reservoir_nodes
import model_validation
import mdp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":
    n_inputs = 1
    system_order = 30
    washout = 30

    inputs, outputs = datasets.narma30(sample_len=1000)
    #data = Dataset({'x':inputs, 'y':outputs})
    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(output_dim=100, input_scaling=0.1)
    readout = RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])
    
    print "Simple training and testing (one fold, i.e. no cross-validation), training_fraction = 0.5."
    print "cross_validate_function = crossvalidation.train_test_only"
    errors = model_validation.validate(data, flow, error_measures.nrmse, cross_validate_function=model_validation.train_test_only, training_fraction=0.5)
    print errors
       
    print "5-fold cross-validation"
    print "cross_validate_function = crossvalidation.cross_validate"
    errors = model_validation.validate(data, flow, error_measures.nrmse, n_folds=5)
    print errors
    print

    print "Leave-one-out cross-validation"
    errors = model_validation.validate(data, flow, error_measures.nrmse, cross_validate_function=model_validation.leave_one_out)
    print errors
