VERBOSE ?= 0

DNEST4_PATH = DNest4/code
EIGEN_PATH = eigen

export CXX = g++

CXXFLAGS = -pthread -std=c++11 -O3 -DNDEBUG -w -DEIGEN_MPL2_ONLY
LIBS = -L$(DNEST4_PATH) -ldnest4 -L/usr/local/lib
includes = -I$(DNEST4_PATH) -I$(EIGEN_PATH) 

SRCDIR = ./src
SRCS =\
$(SRCDIR)/Data.cpp \
$(SRCDIR)/RVConditionalPrior.cpp \
$(SRCDIR)/RVmodel.cpp \
$(SRCDIR)/main.cpp

OBJS=$(subst .cpp,.o,$(SRCS))
HEADERS=$(subst .cpp,.h,$(SRCS))


all: main examples

%.o: %.cpp
ifeq ($(VERBOSE), 0)
	@echo "\033[0;33m Compiling:\033[0m" $<
	@$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)
else
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)
endif


main: $(DNEST4_PATH)/libdnest4.a $(OBJS)
ifeq ($(VERBOSE), 0)
	@echo "\033[0;33m Linking\033[0m "
	@$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)
else
	$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)
endif


.PHONY: examples
examples: $(DNEST4_PATH)/libdnest4.a $(OBJS)
ifeq ($(VERBOSE), 0)
	@make -s -C examples/BL2009
	@echo "\033[0;33m Compiling example\033[0m BL2009"
	@make -s -C examples/CoRoT7
	@echo "\033[0;33m Compiling example\033[0m CoRoT7"
	@make -s -C examples/many_planets
	@echo "\033[0;33m Compiling example\033[0m many_planets"
	@make -s -C examples/default_priors
	@echo "\033[0;33m Compiling example\033[0m default_priors"
else
	@make -C examples/BL2009
	@make -C examples/CoRoT7
	@make -C examples/many_planets
	@make -C examples/default_priors
endif

$(DNEST4_PATH)/libdnest4.a:
ifeq ($(VERBOSE), 0)
	@echo "\033[0;33m Compiling \033[0m DNest4"
	@make -s -C $(DNEST4_PATH) libdnest4.a
else
	make -C $(DNEST4_PATH) libdnest4.a
endif


clean:
	rm -f main $(OBJS)

cleanexamples:
	@make clean -C examples/BL2009
	@make clean -C examples/CoRoT7
	@make clean -C examples/many_planets
	@make clean -C examples/default_priors

cleandnest4:
	@make clean -C $(DNEST4_PATH)

cleanout:
	rm -f sample.txt sample_info.txt levels.txt \
              weights.txt posterior_sample.txt sampler_state.txt \
              posterior_sample_lnlikelihoods.txt

cleanall: cleanout clean cleanexamples cleandnest4
