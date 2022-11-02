# cpp-loadtxt

`numpy.loadtxt` in C++

```c++
#include "loadtxt.hpp"

using namespace std;

int main() {
    auto fname = "data.txt";
    auto data = loadtxt(fname).skiprows(0).comments("#")();

    return 0;
}
```