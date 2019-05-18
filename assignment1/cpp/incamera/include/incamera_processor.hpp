
#ifndef raw_process_hpp
#define raw_process_hpp

#include <stdio.h>
#include <string>
#include <math.h>


class IncameraProcessor {
public:
    // Two basic types for auto white balancing.
    enum AWB_TYPE {
        GREY_WORLD,
        WHITE_WORLD
    };
    IncameraProcessor();
    ~IncameraProcessor();
    //cv::Mat linearization(cv::Mat& in, int min, int max);
    //cv::Mat AWB(cv::Mat& in, AWB_TYPE type);
    //cv::Mat demosaic(cv::Mat& in);
    void process(const char* filename, int rows, int cols);

};

#endif
