KIMA_DIR = ../..

SRC_DIR = $(KIMA_DIR)/src
DNEST4_PATH = $(KIMA_DIR)/DNest4/code
EIGEN_PATH = $(KIMA_DIR)/eigen

includes = -I$(SRC_DIR) -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(KIMA_DIR)/boost

CXXFLAGS = -pthread -std=c++11 -O3 -DNDEBUG -w -DEIGEN_MPL2_ONLY

default_pie := $(shell $(CXX) -v 2>&1 >/dev/null | grep enable-default-pie)
ifneq ($(default_pie),)
  CXXFLAGS += -no-pie
endif

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
	@echo "Cleaning kima outputs  "
	@rm -f sample.txt sample_info.txt levels.txt \
			kima_model_setup.txt \
			weights.txt posterior_sample.txt sampler_state.txt \
			posterior_sample_info.txt

cleanall: clean cleanout
