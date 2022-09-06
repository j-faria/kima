################################################################################
# paths to libraries
################################################################################
DNEST4_PATH = vendor/DNest4/code
INCLUDES = -I$(DNEST4_PATH)

################################################################################
# C++ compiler and compilation flags
################################################################################
CXXFLAGS = -pthread -fPIC -std=c++17 -O3

################################################################################
# libraries to link
################################################################################
LIBS = -L$(DNEST4_PATH) -ldnest4


################################################################################
# files
################################################################################
SRCDIR = kima
SRCS = \
	$(wildcard $(SRCDIR)/distributions/*.cpp)
OBJS = $(SRCS:.cpp=.o)


.PHONY: dnest4
all: main

%.o: %.cpp
	@echo "Compiling:" $<
	@$(CXX) -c $(INCLUDES) -o $@ $< $(CXXFLAGS)

dnest4:
	@echo "Compiling DNest4"
	@+$(MAKE) -s -C $(DNEST4_PATH) libdnest4.a

main: dnest4 $(OBJS)
	@echo "Linking"


################################################################################
# clean-up rules
################################################################################
cleandnest4:
	@echo "Cleaning DNest4"
	@$(MAKE) clean -s -C $(DNEST4_PATH)

clean: cleandnest4
