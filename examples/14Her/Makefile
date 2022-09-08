KIMA_DIR = ../..

DNEST4_PATH = $(KIMA_DIR)/vendor/DNest4/code
EIGEN_PATH = $(KIMA_DIR)/vendor/eigen
LOADTXT_PATH = $(KIMA_DIR)/vendor/cpp-loadtxt/src
INCLUDES = -I$(KIMA_DIR)/kima -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(LOADTXT_PATH)

LIBS = -L$(DNEST4_PATH) -ldnest4

CXXFLAGS = -pthread -std=c++17 -O3 -DNDEBUG -DEIGEN_MPL2_ONLY

kima: kima_setup.cpp
	@$(CXX) -o $@ $< $(CXXFLAGS) $(KIMA_DIR)/kima/libkima.a $(INCLUDES) $(LIBS)


clean:
	rm -f kima kima_setup.o