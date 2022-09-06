#include "loadtxt.hpp"

using namespace std;

int main() {
    {
        auto fname = "data.txt";
        auto data = loadtxt(fname).skiprows(0).comments("#")();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
            data.size(), data[0].size(), fname);
    }
    {
        auto fname = "data.rdb";
        auto data = loadrdb(fname)();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
            data.size(), data[0].size(), fname);
    }
    printf("Done reading!\n");
    return 0;
}