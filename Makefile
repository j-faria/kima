################################################################################
# paths to libraries
################################################################################
DNEST4_PATH = vendor/DNest4/code
EIGEN_PATH = vendor/eigen
LOADTXT_PATH = vendor/cpp-loadtxt/src
INCLUDES = -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(LOADTXT_PATH)

################################################################################
# C++ compiler, linker, and compilation flags
################################################################################
CXXFLAGS = -pthread -fPIC -std=c++17 -O3
LIBTOOL = ar rcs
ifeq ($(shell uname), Darwin)
	LIBTOOL = libtool -static -o
endif


################################################################################
# libraries to link
################################################################################
LIBS = -L$(DNEST4_PATH) -ldnest4


################################################################################
# files
################################################################################
SRCDIR = kima
SRCS = \
	$(wildcard $(SRCDIR)/distributions/*.cpp) \
	$(SRCDIR)/kepler.cpp                      \
	$(SRCDIR)/AMDstability.cpp                \
	$(SRCDIR)/Data.cpp                        \
	$(SRCDIR)/ConditionalPrior.cpp            \
	$(SRCDIR)/RVmodel.cpp

OBJS = $(SRCS:.cpp=.o)


.PHONY: dnest4 main
all: main

%.o: %.cpp
	@echo "Compiling:" $<
	@$(CXX) -c $(INCLUDES) -o $@ $< $(CXXFLAGS)

dnest4:
	@echo "Compiling DNest4"
	@+$(MAKE) -s -C $(DNEST4_PATH) libdnest4.a

main: dnest4 $(OBJS)
	@echo "Linking"
	@$(LIBTOOL) $(SRCDIR)/libkima.a $(OBJS)


################################################################################
# tests
################################################################################
TEST_DIR = tests
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp) $(SRCDIR)/libkima.a
TEST_LIBS = -lgtest -lgtest_main $(LIBS) -L$(SRCDIR) -lkima
TEST_INC = -I$(SRCDIR) $(INCLUDES)

test: main $(TEST_SRCS)
	@echo "Compiling tests"
	@$(CXX) -pthread $(TEST_SRCS) $(TEST_LIBS) $(TEST_INC) -o $(TEST_DIR)/run
	@cd $(TEST_DIR) && ./run


################################################################################
# clean-up rules
################################################################################
cleankima:
	@rm -f $(OBJS)

cleandnest4:
	@echo "Cleaning DNest4"
	@$(MAKE) clean -s -C $(DNEST4_PATH)

clean: cleankima cleandnest4
