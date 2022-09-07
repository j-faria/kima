#include <gtest/gtest.h>
#include "kima.h"
using namespace ::testing;

#include <string>
#include <vector>

// The fixture for testing class RVData
class DataTest : public ::testing::Test {
    protected:
        RVData D;
};


TEST_F(DataTest, LoadNotFound) {
    // Could not read file (data.txt)!
    EXPECT_EXIT(
        D.load("data.txt", "ms"), 
        testing::ExitedWithCode(1), ""
    );
}

TEST_F(DataTest, LoadEmpty) {
    // File seems to be empty (data_tests/empty_file.txt)!
    EXPECT_EXIT(
        D.load("data_tests/empty_file.txt", "ms", 0), 
        testing::ExitedWithCode(1), ""
    );
}

TEST_F(DataTest, LoadOneLine) {
    // read just one line (without newline at end of file)
    D.load("data_tests/just_one_line.txt", "ms", 0);
    EXPECT_EQ(D.N(), 1);
}

TEST_F(DataTest, LoadWithComments) {
    // file with one header line and a comment on one of the lines
    D.load("data_tests/with_comments.txt", "ms", 1);
    EXPECT_EQ(D.N(), 3);
}

TEST_F(DataTest, LoadAndConvert) {
    // read three data points and "convert" from km/s to m/s
    D.load("data_tests/with_comments.txt", "kms", 1);
    EXPECT_EQ(D.N(), 3);
    EXPECT_EQ(D.get_t()[0], 1);  // time should be untouched
    EXPECT_EQ(D.get_y()[0], 2000);
    EXPECT_EQ(D.get_y()[1], 5000);
    EXPECT_EQ(D.get_sig()[0], 3000);
}

TEST_F(DataTest, LoadCSV) {
    // read a csv file
    D.load("data_tests/csv_file.txt", "ms", 1, ",");
    EXPECT_EQ(D.N(), 2);
    EXPECT_DOUBLE_EQ(D.get_t()[0], 1.0);
    EXPECT_DOUBLE_EQ(D.get_y()[0], 2.2);
    EXPECT_DOUBLE_EQ(D.get_y()[1], 5.5);
    EXPECT_DOUBLE_EQ(D.get_sig()[0], 3.0);
}


/* load_multi */
/**************/

TEST_F(DataTest, LoadMulti) {
    // read a multiple instrument file, with 4 columns
    D.load_multi("data_tests/multi_instrument.txt", "ms", 1);
    EXPECT_EQ(D.N(), 10);
    EXPECT_EQ(D.number_instruments, 2);
    EXPECT_EQ(D.get_obsi().size(), D.N());
    EXPECT_EQ(D.get_obsi()[0], 1);
    EXPECT_EQ(D.get_obsi()[1], 1);
    EXPECT_EQ(D.get_obsi()[2], 1);
    EXPECT_EQ(D.get_obsi()[3], 1);
    EXPECT_EQ(D.get_obsi()[4], 1);
    EXPECT_EQ(D.get_obsi()[5], 2);
    EXPECT_EQ(D.get_obsi()[6], 2);
    EXPECT_EQ(D.get_obsi()[7], 2);
    EXPECT_EQ(D.get_obsi()[8], 2);
    EXPECT_EQ(D.get_obsi()[9], 1);
}


TEST_F(DataTest, LoadMultipleFiles) {
    // read multiple files, making sure the times are sorted
    vector<string> datafiles;
    datafiles = {"data_tests/file_inst1.txt", "data_tests/file_inst2.txt"};
    D.load_multi(datafiles, "ms", 2);
    EXPECT_EQ(D.N(), 4);
    EXPECT_EQ(D.number_instruments, 2);
    // 
    EXPECT_DOUBLE_EQ(D.get_t()[0], 1.0); // times should be sorted
    EXPECT_DOUBLE_EQ(D.get_t()[1], 2.0);
    EXPECT_DOUBLE_EQ(D.get_t()[2], 4.0);
    EXPECT_DOUBLE_EQ(D.get_t()[3], 7.0);
    // 
    EXPECT_DOUBLE_EQ(D.get_y()[1], 11.0);   // RVs are (arg)sorted too
    EXPECT_DOUBLE_EQ(D.get_sig()[1], 12.0); // errors are (arg)sorted too
}