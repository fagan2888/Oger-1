import mdp
import Engine

if __name__ == "__main__":
    n_inputs = 1
    system_order = 30
    washout = 30

    inputs, outputs = Engine.datasets.narma30(sample_len=1000)
    #data = Dataset({'x':inputs, 'y':outputs})
    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(output_dim=100, input_scaling=0.1)
    readout = Engine.nodes.RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])
    
    print "Simple training and testing (one fold, i.e. no cross-validation), training_fraction = 0.5."
    print "cross_validate_function = crossvalidation.train_test_only"
    errors = Engine.evaluation.validate(data, flow, Engine.utils.nrmse, cross_validate_function=Engine.evaluation.train_test_only, training_fraction=0.5)
    print errors
       
    print "5-fold cross-validation"
    print "cross_validate_function = crossvalidation.cross_validate"
    errors = Engine.evaluation.validate(data, flow, Engine.utils.nrmse, n_folds=5)
    print errors
    print

    print "Leave-one-out cross-validation"
    errors = Engine.evaluation.validate(data, flow, Engine.utils.nrmse, cross_validate_function=Engine.evaluation.leave_one_out)
    print errors
