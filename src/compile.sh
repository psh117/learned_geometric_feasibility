g++ optimal_bounding_box.cpp $(pkg-config --cflags eigen3) -DCGAL_EIGEN3_ENABLED -lgmp -o ../bin/optimal_bounding_box