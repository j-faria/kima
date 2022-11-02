#include <gtest/gtest.h>
#include "kima.h"
using namespace ::testing;

#include <cmath>

// The fixture for testing class ConditionalPrior
class ConditionalPriorTest : public ::testing::Test {
    protected:
        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, 1, false, RVConditionalPrior());
};


TEST_F(ConditionalPriorTest, PrintPriors) {
    auto c = planets.get_conditional_prior();
    ASSERT_TRUE(bool(c->Pprior));
    ASSERT_TRUE(bool(c->Kprior));
    ASSERT_TRUE(bool(c->eprior));
    ASSERT_TRUE(bool(c->phiprior));
    ASSERT_TRUE(bool(c->wprior));
}


TEST_F(ConditionalPriorTest, LogPDF) {
    auto c = planets.get_conditional_prior();
    ASSERT_TRUE(std::isinf( c->log_pdf({0.0, 0.0, 0.0, 0.0, 0.0}) ));
    ASSERT_TRUE(std::isfinite( c->log_pdf({2.0, 2.0, 0.0, 0.0, 0.0}) ));
}


TEST_F(ConditionalPriorTest, UseHyperpriors) {
    auto c = planets.get_conditional_prior();
    // by default no hyperpriors
    ASSERT_FALSE(bool(c->log_muP_prior));
    ASSERT_FALSE(bool(c->log_muK_prior));
    ASSERT_FALSE(bool(c->wP_prior));

    // turn on hyperpriors
    c->use_hyperpriors();

    ASSERT_TRUE(bool(c->log_muP_prior));
    ASSERT_TRUE(bool(c->log_muK_prior));
    ASSERT_TRUE(bool(c->wP_prior));
}