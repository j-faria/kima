KIMA_DIR = ../..

SRC_DIR = $(KIMA_DIR)/src
DNEST4_PATH = $(KIMA_DIR)/DNest4/code
EIGEN_PATH = $(KIMA_DIR)/eigen
CELERITE_PATH = $(KIMA_DIR)/celerite/cpp/include

includes = -I$(SRC_DIR) -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(CELERITE_PATH)

CXX = g++
CXXFLAGS = -pthread -std=c++11 -O3 -w
LIBS = -ldnest4 -L/usr/local/lib

KIMA_SRCS =\
$(SRC_DIR)/MyConditionalPrior.cpp \
$(SRC_DIR)/Data.cpp \
kima_setup.cpp \
$(SRC_DIR)/MyModel.cpp

KIMA_OBJS = $(subst .cpp,.o,$(KIMA_SRCS))

%.o: %.cpp
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)

kima: $(KIMA_OBJS)
	$(CXX) -o kima $(KIMA_OBJS) -L$(DNEST4_PATH) $(LIBS) $(CXXFLAGS)

clean:
	rm -f kima_setup.o kima

cleanall: clean
	rm -f sample.txt sample_info.txt sampler_state.txt
