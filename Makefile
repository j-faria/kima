DNEST4_PATH = DNest4/code
EIGEN_PATH = eigen
CELERITE_PATH = celerite/cpp/include

#export CXX = g++

CXXFLAGS = -pthread -std=c++11 -O3 -DNDEBUG -w -DEIGEN_MPL2_ONLY

default_pie := $(shell $(CXX) -v 2>&1 >/dev/null | grep enable-default-pie)
ifneq ($(default_pie),)
  CXXFLAGS += -no-pie
endif

LIBS = -L$(DNEST4_PATH) -ldnest4 -L/usr/local/lib
includes = -I$(DNEST4_PATH) -I$(EIGEN_PATH) -I$(CELERITE_PATH)

SRCDIR = ./src
SRCS =\
$(wildcard $(SRCDIR)/distributions/*.cpp) \
$(SRCDIR)/Data.cpp \
$(SRCDIR)/RVConditionalPrior.cpp \
$(SRCDIR)/RVmodel.cpp \
$(SRCDIR)/main.cpp

OBJS=$(subst .cpp,.o,$(SRCS))
HEADERS=$(subst .cpp,.h,$(SRCS))

EXAMPLES = 51Peg BL2009 CoRoT7 many_planets multi_instrument trends \
           activity_correlations default_priors studentT arbitrary_units K2-24

all: main ${EXAMPLES}

%.o: %.cpp
	@echo "Compiling:" $<
	@$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)


main: $(DNEST4_PATH)/libdnest4.a $(OBJS)
	@echo "Linking"
	@$(CXX) -o kima $(OBJS) $(LIBS) $(CXXFLAGS)


.PHONY: ${EXAMPLES}
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
	@+$(MAKE) -s -C $(DNEST4_PATH) libdnest4.a


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
