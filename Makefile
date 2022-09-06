################################################################################
# paths to libraries
################################################################################
DNEST4_PATH = vendor/DNest4/code


################################################################################
# C++ compiler and compilation flags
################################################################################
CXXFLAGS = -pthread -fPIC -std=c++17 -O3

################################################################################
# libraries
################################################################################
LIBS = -L$(DNEST4_PATH) -ldnest4

.PHONY: dnest4

dnest4:
	@echo "Compiling DNest4"
	@+$(MAKE) -s -C $(DNEST4_PATH) libdnest4.a

cleandnest4:
	@echo "Cleaning DNest4"
	@$(MAKE) clean -s -C $(DNEST4_PATH)

clean: cleandnest4
