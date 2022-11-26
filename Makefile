################################################################################
# paths to libraries
################################################################################
DNEST4_PATH = kima/vendor/DNest4/code
EIGEN_PATH = kima/vendor/eigen
LOADTXT_PATH = kima/vendor/cpp-loadtxt/src
INCLUDES = -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(LOADTXT_PATH)

################################################################################
# C++ compiler, linker, and compilation flags
################################################################################
CXXFLAGS = -pthread -std=c++17 -O3
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
	$(SRCDIR)/GP.cpp                          \
	$(SRCDIR)/RVmodel.cpp                     \
	$(SRCDIR)/GPmodel.cpp                     \
	$(SRCDIR)/GPmodel2.cpp                    \
	$(SRCDIR)/RVFWHMmodel.cpp                 \
	$(SRCDIR)/BDmodel.cpp                     \
	$(SRCDIR)/BINARIESmodel.cpp

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


EXAMPLES = 14Her Kepler16

${EXAMPLES}: main
	@echo "Compiling example $@"
	@$(MAKE) -s -C kima/examples/$@;

examples: $(EXAMPLES)

################################################################################
# run examples
################################################################################
run_examples: ${EXAMPLES}
	@echo "Running examples"
	@+for example in $(EXAMPLES) ; do \
		echo "Running $$example"; \
		cd kima/examples/$$example && ./kima && cd ../../.. ; \
	done

################################################################################
# tests
################################################################################
THIS_DIR = $(shell pwd)
TEST_DIR = tests
# TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_SRCS = $(TEST_DIR)/test_Data.cpp
TEST_LIBS = -lgtest -lgtest_main $(LIBS) #-L$(SRCDIR) -lkima
TEST_INC = -I$(SRCDIR) $(INCLUDES)

test: main $(TEST_SRCS)
	@echo "Compiling tests"
	@+for test_src in $(TEST_SRCS) ; do \
		$(CXX) $(CXXFLAGS) $$test_src $(SRCDIR)/libkima.a $(TEST_LIBS) $(TEST_INC) -o $(TEST_DIR)/run ; \
		cd $(TEST_DIR) && ./run && cd $(THIS_DIR) ; \
	done


################################################################################
# clean-up rules
################################################################################
cleankima:
	@echo "Cleaning kima"
	@rm -f $(OBJS) $(SRCDIR)/libkima.a

cleandnest4:
	@echo "Cleaning DNest4"
	@$(MAKE) clean -s -C $(DNEST4_PATH)

cleantest:
	@rm -f $(TEST_DIR)/run

cleanexamples:
	@+for example in $(EXAMPLES) ; do \
		echo "Cleaning example $$example"; \
		$(MAKE) clean -s -C kima/examples/$$example; \
	done

clean: cleankima cleandnest4 cleantest cleanexamples
