#include "incamera_processor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int read_file( char const* src_filename, unsigned char*& buf )
{
    if (buf == nullptr)
        return 0;
    FILE *fp = fopen(src_filename, "rb");
    if(!fp)
        assert(false && "fopen file failed!");
     fseek(fp, 0, SEEK_END);
     int size = ftell(fp);
     if (buf != nullptr) {
        fseek(fp, 0, SEEK_SET);
        buf = new unsigned char[size];
        fread(buf, size, 1, fp);
    }
    fclose(fp);
    return size;
}

void dump_float(const char* file, cv::Mat& img) {
    cv::Mat out(img.rows, img.cols, CV_MAKETYPE(CV_8U, img.channels()));
    float* pin = (float*) img.data; //new float[cols*rows];
    uint8_t* pout = (uint8_t*) out.data; //new float[cols*rows];
    int total = img.rows * img.cols * img.channels();
    for (int i = 0; i < total; ++i) {
        if (i % 1000000 == 0) {
            //cout<<pin[i]<<endl;
        }
        pout[i] = (uint8_t)(pin[i] * 255.0);
    }
    if (img.channels() == 3) {
       cv::Mat rgb2bgr;
       cvtColor(out, rgb2bgr, COLOR_RGB2BGR);
       cv::imwrite(file, rgb2bgr);
    } else {
        cv::imwrite(file, out);
    }
}

cv::Mat read_raw(const char* src_filename)
{
    //unsigned char* buf = new unsigned char;
    //int size = read_file(src_filename, buf);
    //unsigned short* image = (unsigned short*)buf;
    //Mat out(cv::Size(cols, rows), CV_32FC1);
    //float* pf =  (float*) out.data; //new float[cols*rows];
    //int max = 0;
    //for (int i=0; i<rows; i++){
        //for(int j=0; j<cols; j++){
            //int idx = i*cols + j;
            //pf[idx] = float(image[idx]);
            //max = image[idx] > max ? image[idx] : max;
        //}
    //}
    //cout<<"max:"<<max<<endl;
    //delete buf;
    //buf = NULL;
    //pf = NULL;
    //image = NULL;
    //return out;

    Mat img = cv::imread(src_filename, -1);
    cout<<"sss"<<img.cols << "  " <<img.rows<<endl;

    //int* pf =  (int*) out.data; //new float[cols*rows];
    //int max = 0;
    //for (int i=0; i<rows; i++){
        //for(int j=0; j<cols; j++){
            //int idx = i*cols + j;
            //max = pf[idx] > max ? pf[idx] : max;
            //pf[idx] = float(image[idx]);
        //}
    //}
    //cout<<"max:"<<max<<endl;
    cv::Mat out(img.rows, img.cols, CV_MAKETYPE(CV_8U, img.channels()));
    //Mat out();
    img.convertTo(out, CV_32F);
    return out;
}

IncameraProcessor::IncameraProcessor() {
    printf("Constructing incamera processor. \n");
}

IncameraProcessor::~IncameraProcessor() {
    printf("Deconstructing incamera processor. \n");
}

cv::Mat linearization(cv::Mat& in, int min, int max) {
    printf("linearization. \n");
    cv::Mat out = in.clone();
    int range = max - min + 1;
    float tmp = 0.f;
    float* pf =  (float*) out.data; //new float[cols*rows];
    int count = 0;
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {
            int idx = i * out.cols + j;
            tmp = pf[idx];
            tmp = (tmp - min) / range;
            if (tmp < 0) {
                count++;
                tmp = 0;
            }
            else if (tmp > 1) {
                tmp = 1;
            }
            pf[idx] = tmp;
        }
    }
    return out;
}

//cv::Mat vanilla_filter(cv::Mat& in, cv::Mat& kernel) {
    //int w ,
//}



cv::Mat AWB(cv::Mat& in, IncameraProcessor::AWB_TYPE mode) {
    cv::Mat out = in.clone();
    double r_r = 0.f, r_g = 0.f, r_b = 0.f;
    float* pin =  (float*)in.data;
    if (mode == IncameraProcessor::WHITE_WORLD) {
        for (int i = 0; i < in.rows; ++i) {
            for (int j = 0; j < in.cols; ++j) {
                int idx = i * in.cols + j;
                if (i % 2 == 0) {
                    if (j % 2 == 0) {
                        r_r = std::max(r_r, (double)pin[idx]);
                    } else {
                        r_g = std::max(r_g, (double)pin[idx]);
                    }
                } else {
                    if (j % 2 == 0) {
                        r_g = std::max(r_g, (double)pin[idx]);
                    } else {
                        r_b = std::max(r_b, (double)pin[idx]);
                    }
                }
            }
        }
    } else if (mode == IncameraProcessor::GREY_WORLD) {
        double r_sum = 0.f, g_sum = 0.f, b_sum = 0.f;
        for (int i = 0; i < in.rows; ++i) {
            for (int j = 0; j < in.cols; ++j) {
                int idx = i * in.cols + j;
                if (i % 2 == 0) {
                    if (j % 2 == 0) {
                        r_sum += pin[idx];
                    } else {
                        g_sum += pin[idx];
                    }
                } else {
                    if (j % 2 == 0) {
                        g_sum += pin[idx];
                    } else {
                        b_sum += pin[idx];
                    }
                }
            }
        }
        size_t len = in.rows * in.cols;
        r_r = r_sum * 4 / len;
        r_g = g_sum * 2 / len;
        r_b = b_sum * 4 / len;
    }
    float* pout = (float*)out.data;
    for (int i = 0; i < in.rows; ++i) {
        for (int j = 0; j < in.cols; ++j) {
            int idx = i * in.cols + j;
            if (i % 2 == 0) {
                if (j % 2 == 0) {
                    pout[idx] = (float)r_g * pin[idx] / (float)r_r;
                }
            } else {
                if (j % 2 != 0) {
                    pout[idx] = (float)r_g * pin[idx] / (float)r_b;
                }
            }
        }
    }
    return out;
}

cv::Mat demosaic(cv::Mat& in) {
    // rggb
    int h = in.rows, w = in.cols, c = in.channels();
    cv::Mat r = Mat::zeros(in.size(), CV_32FC1);
    cv::Mat g = Mat::zeros(in.size(), CV_32FC1);
    cv::Mat b = Mat::zeros(in.size(), CV_32FC1);
    float* r_ptr = (float*) r.data;
    float* g_ptr = (float*) g.data;
    float* b_ptr = (float*) b.data;
    float* pin =  (float*)in.data;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            size_t idx = i * w + j;
            if (i % 2 == 0) {
                if (j % 2 == 0) {
                    r_ptr[idx] = pin[idx];
                } else {
                    g_ptr[idx] = pin[idx];
                }
            } else {
                if (j % 2 == 0) {
                    g_ptr[idx] = pin[idx];
                } else {
                    b_ptr[idx] = pin[idx];
                }
            }
        }
    }

    cv::Mat out(h, w, CV_MAKETYPE(CV_32F, 3));
    float* pout = (float*)out.data;
    float r_v = 0.f, g_v = 0.f, b_v = 0.f;
    float g_sum, g_count, r_sum, r_count, b_sum, b_count;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            size_t idx = i * w + j;
            if (i % 2 == 0) {
                // odd row
                if (j % 2 != 0) {
                    // keep g
                    r_sum = r_count = 0.0;
                    if (j + 1 < w) {
                        r_sum += r_ptr[i * w +  (j + 1)];
                        r_count += 1.0;
                    }
                    if (j - 1 >= 0) {
                        r_sum += r_ptr[i * w +  (j - 1)];
                        r_count += 1.0;
                    }
                    b_sum = b_count = 0.0;
                    if (i + 1 < h) {
                        b_sum += b_ptr[(i + 1) * w + j];
                        b_count += 1.0;
                    }
                    if (i - 1 >= 0) {
                        b_sum += b_ptr[(i - 1) * w + j];
                        b_count += 1.0;
                    }
                    r_v = (float)(r_sum / r_count);

                    b_v = b_sum / (float)b_count;
                    g_v = g_ptr[idx];

                } else {
                    // keep r
                    g_sum = g_count = 0.0;
                    if (i - 1 >= 0) {
                        g_sum += g_ptr[(i - 1) * w +  j];
                        g_count += 1.0;
                    }
                    if (j + 1 < w) {
                        g_sum += g_ptr[i * w +  (j + 1)];
                        g_count += 1.0;
                    }
                    if (j - 1 >= 0) {
                        g_sum += g_ptr[i * w +  (j - 1)];
                        g_count += 1.0;
                    }
                    if (i + 1 < h) {
                        g_sum += g_ptr[(i + 1) * w + j];
                        g_count += 1.0;
                    }
                    b_sum = b_count = 0.0;
                    if (i - 1 >= 0 && j - 1 >= 0) {
                        b_sum += b_ptr[(i - 1) * w +  j - 1];
                        b_count += 1.0;
                    }
                    if (i - 1 >= 0 && j + 1 < w) {
                        b_sum += b_ptr[(i - 1) * w +  (j + 1)];
                        b_count += 1.0;
                    }
                    if (i + 1 < h && j - 1 >= 0) {
                        b_sum += b_ptr[(i + 1) * w +  (j - 1)];
                        b_count += 1.0;
                    }
                    if (i + 1 < h && j + 1 < w) {
                        b_sum += b_ptr[(i + 1) * w +  (j + 1)];
                        b_count += 1.0;
                    }
                    g_v = g_sum / (float)g_count;
                    b_v = b_sum / (float)b_count;
                    r_v = r_ptr[idx];
                }
            } else {
                if (j % 2 == 0) {
                    // keep g
                    r_sum = r_count = 0.0;
                    b_sum = b_count = 0.0;
                    if (j + 1 < w) {
                        b_sum += b_ptr[i * w +  (j + 1)];
                        b_count += 1.0;
                    }
                    if (j - 1 >= 0) {
                        b_sum += b_ptr[i * w +  (j - 1)];
                        b_count += 1.0;
                    }
                    if (i + 1 < h) {
                        r_sum += r_ptr[(i + 1) * w + j];
                        r_count += 1.0;
                    }
                    if (i - 1 >= 0) {
                        r_sum += r_ptr[(i - 1) * w + j];
                        r_count += 1.0;
                    }
                    r_v = r_sum / r_count;
                    b_v = b_sum / b_count;
                    g_v = g_ptr[idx];
                } else {
                    // keep b
                    g_sum = g_count = 0.0;
                    if (i - 1 >= 0) {
                        g_sum += g_ptr[(i - 1) * w +  j];
                        g_count += 1.0;
                    }
                    if (i + 1 < h) {
                        g_sum += g_ptr[(i + 1) * w + j];
                        g_count += 1.0;
                    }
                    if (j + 1 < w) {
                        g_sum += g_ptr[i * w +  (j + 1)];
                        g_count += 1.0;
                    }
                    if (j - 1 >= 0) {
                        g_sum += g_ptr[i * w +  (j - 1)];
                        g_count += 1.0;
                    }
                    r_sum = r_count = 0.0;
                    if (i - 1 >= 0 && j - 1 >= 0) {
                        r_sum += r_ptr[(i - 1) * w +  j - 1];
                        r_count += 1.0;
                    }
                    if (i - 1 >= 0 && j + 1 < w) {
                        r_sum += r_ptr[(i - 1) * w +  (j + 1)];
                        r_count += 1.0;
                    }
                    if (i + 1 < h && j - 1 >= 0) {
                        r_sum += b_ptr[(i + 1) * w +  (j - 1)];
                        r_count += 1.0;
                    }
                    if (i + 1 < h && j + 1 < w) {
                        r_sum += r_ptr[(i + 1) * w +  (j + 1)];
                        r_count += 1.0;
                    }
                    g_v = g_sum / g_count;
                    r_v = r_sum / r_count;
                    b_v = b_ptr[idx];
                }
            }
            pout[idx * 3] = r_v;
            pout[idx * 3 + 1] = g_v;
            pout[idx * 3 + 2] = b_v;
        }
    }
    return out;
}

cv::Mat bright(cv::Mat& in, float scale) {
    cv::Mat out = in.clone();
    float* pout = (float*)out.data;
    int h = in.cols, w = in.rows, c = in.channels();
    int len = h * w * c;
    for (int i = 0; i < len; ++i) {
        pout[i] *= scale;
    }
    return out;
}

cv::Mat gamma(cv::Mat& in) {
    cv::Mat out = in.clone();
    float* pout = (float*)out.data;
    int h = in.cols, w = in.rows, c = in.channels();
    int len = h * w * c;
    for (int i = 0; i < len; ++i) {
        if (pout[i] <= 0.0031308) {
            pout[i] *= 12.92;
        } else {
            pout[i] = std::pow(pout[i], (1.0 / 2.4)) * (1 + 0.055) - 0.055;
        }
    }
    return out;

}

void IncameraProcessor::process(char const* filename, int rows, int cols) {
    printf("Processing with: %s. \n", filename);
    cv::Mat img = read_raw(filename);
    img = linearization(img, 2047, 15000);
    dump_float("test1.png", img);
    img = AWB(img, WHITE_WORLD);
    //dump_float("test2.png", img);
    //img = AWB(img, GREY_WORLD);
    dump_float("awb.png", img);
    img = demosaic(img);
    dump_float("demosaic.png", img);
    img = bright(img, 1.84);
    dump_float("bright.png", img);
    img = gamma(img);
    dump_float("result.png", img);
}
