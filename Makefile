# Put the directory of 'DNest4/code' into this variable
DNEST4_PATH = DNest4/code

EIGEN_PATH = eigen

# CELERITE_PATH = celerite/cpp/include

includes = -I$(DNEST4_PATH) -I$(EIGEN_PATH) 
# -I$(CELERITE_PATH)

CXX = g++
CXXFLAGS = -pthread -std=c++11 -O3 -DNDEBUG -w -DEIGEN_MPL2_ONLY
LIBS = -ldnest4 -L/usr/local/lib


SRCDIR = ./src
SRCS =\
$(SRCDIR)/Data.cpp \
$(SRCDIR)/RVConditionalPrior.cpp \
$(SRCDIR)/RVmodel.cpp \
$(SRCDIR)/main.cpp

OBJS=$(subst .cpp,.o,$(SRCS))
HEADERS=$(subst .cpp,.h,$(SRCS))

all: main pythoncheck

%.o: %.cpp
	$(CXX) -c $(includes) -o $@ $< $(CXXFLAGS)


main: $(DNEST4_PATH)/libdnest4.a $(OBJS)
	$(CXX) -o main $(OBJS) -L$(DNEST4_PATH) $(LIBS) $(CXXFLAGS)

$(DNEST4_PATH)/libdnest4.a:
	make noexamples -C $(DNEST4_PATH)

pythoncheck:
	$(eval NOPACK=0)
	@python -c "import numpy" 2>/dev/null ; \
		if [ $$? -eq 1 ] ; \
		then $(eval $NOPACK=1) echo "numpy does not seem to be installed!"; fi

	@python -c "import scipy" 2>/dev/null ; \
		if [ $$? -eq 1 ] ; \
		then $(eval $NOPACK=1) echo "scipy does not seem to be installed!"; fi

	@python -c "import pandas" 2>/dev/null ; \
		if [ $$? -eq 1 ] ; \
		then $(eval $NOPACK=1) echo "pandas does not seem to be installed!"; fi

	@python -c "import matplotlib" 2>/dev/null ; \
		if [ $$? -eq 1 ] ; \
		then $(eval $NOPACK=1) echo "matplotlib does not seem to be installed!"; fi

	@python -c "import corner" 2>/dev/null ; \
		if [ $$? -eq 1 ] ; \
		then $(eval $NOPACK=1) echo "corner does not seem to be installed!"; fi

	@if [ $(NOPACK) -eq 1 ] ; \
		then echo \
			"Some required Python packages are missing," \
			"kima will still work but not the analysis of results.";\
		fi



clean:
	rm -f main $(OBJS)

cleanout:
	rm -f sample.txt sample_info.txt levels.txt weights.txt posterior_sample.txt sampler_state.txt

cleanall: clean
	rm -f sample.txt sample_info.txt \
		levels.txt weights.txt posterior_sample.txt sampler_state.txt \
		posterior_sample_lnlikelihoods.txt
