VERBOSE ?= 1

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

EXAMPLES = BL2009 CoRoT7 many_planets 51Peg default_priors

all: main examples

%.o: %.cpp
ifeq ($(VERBOSE), 1)
	@echo "\033[0;33m Compiling:\033[0m" $<
	@$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)
else
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)
endif


main: $(DNEST4_PATH)/libdnest4.a $(OBJS)
ifeq ($(VERBOSE), 1)
	@echo "\033[0;33m Linking\033[0m "
	@$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)
else
	$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)
endif


.PHONY: examples
examples: $(DNEST4_PATH)/libdnest4.a $(OBJS)
ifeq ($(VERBOSE), 1)
	@for example in $(EXAMPLES) ; do \
		echo "\033[0;33m Compiling example\033[0m $$example"; \
		make -s -C examples/$$example; \
	done
else
	@for example in $(EXAMPLES) ; do \
		make -s -C examples/$$example; \
	done 
endif

$(DNEST4_PATH)/libdnest4.a:
ifeq ($(VERBOSE), 1)
	@echo "\033[0;33m Compiling \033[0m DNest4"
	@make -s -C $(DNEST4_PATH) libdnest4.a
else
	make -C $(DNEST4_PATH) libdnest4.a
endif


clean:
	@rm -f kima $(OBJS)

cleanexamples:
ifeq ($(VERBOSE), 1)
	@for example in $(EXAMPLES) ; do \
		echo "\033[0;33m Cleaning example \033[0m $$example"; \
		make clean -s -C examples/$$example; \
	done
else
	@for example in $(EXAMPLES) ; do \
		make clean -s -C examples/$$example; \
	done
endif

cleandnest4:
ifeq ($(VERBOSE), 1)
	@echo "\033[0;33m Cleaning \033[0m DNest4"
endif
	@make clean -s -C $(DNEST4_PATH)

cleanout:
ifeq ($(VERBOSE), 1)
	@echo "\033[0;33m Cleaning kima outputs \033[0m "
endif
	@rm -f sample.txt sample_info.txt levels.txt \
			kima_model_setup.txt \
			weights.txt posterior_sample.txt sampler_state.txt \
			posterior_sample_lnlikelihoods.txt

cleanall: cleanout clean cleanexamples cleandnest4
