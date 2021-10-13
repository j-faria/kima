DNEST4_PATH = DNest4/code
EIGEN_PATH = eigen
CELERITE_PATH = celerite/cpp/include
STATS_PATH = vendor/stats/include
GCEM_PATH = vendor/gcem/include

# export CXX = g++
# export CXX = clang++-9

CXXFLAGS = -pthread -fPIC -std=c++17 -O3 -DNDEBUG -w -DEIGEN_MPL2_ONLY -Wfatal-errors
CXXFLAGS += -Wno-inconsistent-missing-override
CXXFLAGS += -g

default_pie := $(shell $(CXX) -v 2>&1 >/dev/null | grep enable-default-pie)
ifneq ($(default_pie),)
  CXXFLAGS += -no-pie
endif

LIBS = -L$(DNEST4_PATH) -ldnest4 -L/usr/local/lib
includes = -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(CELERITE_PATH) -I$(STATS_PATH) -I$(GCEM_PATH)

# for modules and functions that use pybind11
pybind_includes := `python3 -m pybind11 --includes`
# includes += $(pybind_includes)


SRCDIR = ./src
SRCS =\
$(wildcard $(SRCDIR)/distributions/*.cpp) \
$(SRCDIR)/Data.cpp \
$(SRCDIR)/kepler.cpp \
$(SRCDIR)/AMDstability.cpp \
$(SRCDIR)/ConditionalPrior.cpp \
$(SRCDIR)/RVmodel.cpp \
$(SRCDIR)/RVFWHMmodel.cpp \
$(SRCDIR)/main.cpp
# $(SRCDIR)/WFmodel.cpp \
# $(SRCDIR)/DataPy.cpp \

OBJS=$(subst .cpp,.o,$(SRCS))
HEADERS=$(subst .cpp,.h,$(SRCS))

EXAMPLES = 51Peg BL2009 CoRoT7 many_planets multi_instrument trends \
           activity_correlations default_priors studentT arbitrary_units K2-24

all: main ${EXAMPLES}

%.o: %.cpp
	@echo "Compiling:" $<
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)


main: $(DNEST4_PATH)/libdnest4.a $(OBJS)
	@echo "Linking"
	@$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)


.PHONY: ${EXAMPLES} bench test

# old way, does not respect make -j flag
# examples: $(DNEST4_PATH)/libdnest4.a $(OBJS)
# 	@+for example in $(EXAMPLES) ; do \
# 		echo "Compiling example $$example" & $(MAKE) -s -C examples/$$example; \
# 	done

${EXAMPLES}: $(DNEST4_PATH)/libdnest4.a $(OBJS)
	@echo "Compiling example $@"
	@$(MAKE) -s -C examples/$@;

$(DNEST4_PATH)/libdnest4.a:
	@echo "Compiling DNest4"
	+$(MAKE) -s -C $(DNEST4_PATH) libdnest4.a



pyAMD:
	$(CXX) -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/AMDstability.o src/pyAMDstability.cpp -o pykima/pyAMD`python3-config --extension-suffix`

pykepler: src/kepler.o src/pykepler.cpp
	$(CXX) $(CXXFLAGS) -shared -I. $(pybind_includes) $^ -o pykima/pykepler`python3-config --extension-suffix`


test:
	@make -C tests
	@pytest tests --disable-pytest-warnings

bench:
	@$(MAKE) -C benchmarks


clean:
	@rm -f kima $(OBJS)

cleanexamples:
	@+for example in $(EXAMPLES) ; do \
		echo "Cleaning example $$example"; \
		$(MAKE) clean -s -C examples/$$example; \
	done

cleandnest4:
	@echo "Cleaning DNest4"
	@$(MAKE) clean -s -C $(DNEST4_PATH)

cleanout:
	@echo "Cleaning kima outputs  "
	@rm -f sample.txt sample_info.txt levels.txt \
			kima_model_setup.txt \
			weights.txt posterior_sample.txt sampler_state.txt \
			posterior_sample_info.txt

cleanall: cleanout clean cleanexamples cleandnest4
