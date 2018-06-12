KIMA_DIR = ../..

SRC_DIR = $(KIMA_DIR)/src
DNEST4_PATH = $(KIMA_DIR)/DNest4/code
EIGEN_PATH = $(KIMA_DIR)/eigen
CELERITE_PATH = $(KIMA_DIR)/celerite/cpp/include

includes = -I$(SRC_DIR) -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(CELERITE_PATH)

CXXFLAGS = -pthread -std=c++11 -O3 -w
LIBS = -ldnest4 -L/usr/local/lib

KIMA_SRCS =\
$(SRC_DIR)/RVConditionalPrior.cpp \
$(SRC_DIR)/Data.cpp \
$(SRC_DIR)/RVmodel.cpp \
kima_setup.cpp

KIMA_OBJS = $(subst .cpp,.o,$(KIMA_SRCS))

%.o: %.cpp
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)

kima: $(KIMA_OBJS)
	$(CXX) -o kima $(KIMA_OBJS) -L$(DNEST4_PATH) $(LIBS) $(CXXFLAGS)

clean:
	rm -f kima_setup.o kima

cleanout:
	rm -f sample.txt sample_info.txt \
		levels.txt weights.txt posterior_sample.txt sampler_state.txt \
		posterior_sample_lnlikelihoods.txt

cleanall: clean cleanout
