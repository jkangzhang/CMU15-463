#include <cstdio>
#include <stdio.h>
#include "incamera_processor.hpp"



int main(int argc, char** argv) {
    printf("main......\n");
    const char* img_path = argv[1];
    IncameraProcessor processor;
    processor.process(img_path, 2856, 4290);
    return 0;
}
