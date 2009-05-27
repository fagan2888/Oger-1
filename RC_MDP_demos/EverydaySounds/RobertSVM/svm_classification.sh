#!/bin/sh
#===============================================================================
# script for LIBSVM classification
#===============================================================================

# train data
#DATA=../data/drip_vs_flow_mat.libsvm
#DATA=../data/drip_vs_flow_dat.libsvm

DATA=../data/wind_mfccint_vs_all_dat.libsvm

#SOUND_LISTS = ["deformation.list", "explosion.list", "friction.list", "pour.list",
#               "whoosh.list", "drip.list", "flow.list", "impact.list",
#               "rolling.list", "wind.list"]


# n-fold cross validation
CROSSVAL=10

# SVM Parameters (prefered to run cross validation !)
# C:
#SVM_C=2.0      # for scaled
#SVM_C=8192.0      # for unscaled
# gamma:
#SVM_gamma=0.125      # for scaled
#SVM_gamma=0.00048828125      # for unscaled

# add path of libsvm to PATH
LIBSVM_PATH=/home/holzi/versioniert/libsvm-2.88
PATH=$PATH:$LIBSVM_PATH

# path to grid.py (for grid search)
GRID_PY=$LIBSVM_PATH/tools/grid.py


#===============================================================================
# some functions

unscaled_validation ()
{
	# make crossvalidation
	svm-train -v $CROSSVAL $DATA
}

scaled_validation ()
{
	# first scale data
	svm-scale -l -1 -u 1 $DATA > tmp.dat
	
	# make crossvalidation
	svm-train -v $CROSSVAL tmp.dat
	
	# remove scaled data
	rm tmp.dat
}

unscaled_gridsearch ()
{
	# gridsearch
	$GRID_PY -svmtrain $LIBSVM_PATH/svm-train -v $CROSSVAL $DATA

	# remove tmp data
	rm *.out
	rm *.png
}

scaled_gridsearch ()
{
	# scale data
	svm-scale -l -1 -u 1 $DATA > tmp.dat

	# gridsearch
	$GRID_PY -svmtrain $LIBSVM_PATH/svm-train -v $CROSSVAL tmp.dat

	# remove tmp data
	rm tmp.dat
	rm *.out
	rm *.png
}




#===============================================================================
# MAIN

# 1. with grid search:
#unscaled_gridsearch
scaled_gridsearch

# 2. without cross validation (only for testing):
#unscaled_validation
#scaled_validation

