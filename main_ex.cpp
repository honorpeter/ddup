//
// Created by Tom Wang on 2018/12/18.
//

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <unistd.h>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <format_reader_ptr.h>

#include <opencv2/opencv.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <pthread.h>
#include <sched.h>
#include<ctype.h>

#include "classification_sample.h"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

inline float sub_mean(cv::Mat &image, float *mean_arr, int x, int y, int width, int c, int mean_delta_a) {
    unsigned char r = image.at<cv::Vec3b>(y, x)[c];
    float mean_r = mean_arr[y * width + x + mean_delta_a * width * width];
    return (r - mean_r) / 255.0f;
}

void createPlugin(InferencePlugin &plugin) {

    InferenceEnginePluginPtr engine_ptr = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getSuitablePlugin(
            TargetDevice::eCPU);
    plugin = InferencePlugin(engine_ptr);
    plugin.SetConfig({{PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES}});
    printPluginVersion(plugin, std::cout);
}

void print_head_from_arr(float *head, int size) {
    for (int j = 0; j < size; ++j) {
        slog::info << " " << *(head + j);
        if (j % 3 == 2) {
            slog::info << slog::endl;
        }
    }
    slog::info << slog::endl;
}

void print_head_from_arr(unsigned char *head, int size) {
    for (int j = 0; j < size; ++j) {
        printf("%hhu ", *(head + j));
    }
    printf("\n ");
}

void print_image_head(cv::Mat &image, int size) {
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            printf("%d_%d: %hhu  \n", y, x, image.at<cv::Vec3b>(y, x)[0]);
            print_head_from_arr(&image.at<cv::Vec3b>(y, x)[0], 3);
        }
    }
}


void readNet(CNNNetReader &networkReader) {
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";

    networkReader.ReadNetwork(FLAGS_m);
    networkReader.ReadWeights(binFileName);
    CNNNetwork network = networkReader.getNetwork();

    slog::info << "Preparing input blobs" << slog::endl;
    InputsDataMap inputInfo = network.getInputsInfo();
    if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
    auto inputInfoItem = *inputInfo.begin();
    inputInfoItem.second->setPrecision(Precision::FP32);
    inputInfoItem.second->setLayout(Layout::NHWC);
    network.setBatchSize(8);

    slog::info << "Preparing output blobs" << slog::endl;
    OutputsDataMap outputInfo(network.getOutputsInfo());

    std::string firstOutputName;
    for (auto &item : outputInfo) {
        if (firstOutputName.empty()) {
            firstOutputName = item.first;
        }
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }
        outputData->setPrecision(Precision::FP32);
        outputData->setLayout(Layout::NC);
    }
}

void *run(void *p) {

    InferRequest *infer_request = (InferRequest *) p;

    // --------------------------- 7. Do inference ---------------------------------------------------------
    slog::info << "Starting inference (" << FLAGS_ni << " iterations)" << slog::endl;

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    double total = 0.0;
    /** Start inference & calc performance **/
    for (int iter = 0; iter < FLAGS_ni; ++iter) {
        auto t0 = Time::now();
        infer_request->Infer();
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total += d.count();
    }

    // -----------------------------------------------------------------------------------------------------
    std::cout << std::endl << "total inference time: " << total << std::endl;
    std::cout << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms"
              << std::endl;
    std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * 8 / total << " FPS"
              << std::endl;
    std::cout << std::endl;

    return 0;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than zero (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void fill_image_2_arr(float *phead, cv::Mat &image, int offset) {

    slog::info << "Star fill data offset#" << offset << slog::endl;

    int delta_green = 224 * 224;
    int delta_blue = 224 * 224 * 2;

    for (int i = 0; i < image.rows; ++i) {
        for (int z = 0; z < image.cols; ++z) {
            *(phead + offset) = image.at<cv::Vec3f>(i, z)[0];
            *(phead + offset + delta_green) = image.at<cv::Vec3f>(i, z)[1];
            *(phead + offset + delta_blue) = image.at<cv::Vec3f>(i, z)[2];
        }
    }
}

inline void crop(const float *psrc, float *&pdst, int x_offset, int y_offset, int width, int height, int debug) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width * 3; x += 3) {
            int r_index = y * width * 3 + x;
            int g_index = r_index + 1;
            int b_index = r_index + 2;
            int r_offset_index = (y + y_offset) * 256 + x/3 + x_offset;
            int g_offset_index = (y + y_offset) * 256 + x/3 + x_offset + 256 * 256;
            int b_offset_index = (y + y_offset) * 256 + x/3 + x_offset + 256 * 256 * 2;
            if (y == 0 && debug) {
                printf("x:%d rsrc_dst_value:%d_%d_%f\n", x / 3, r_index, r_offset_index, *(psrc + r_offset_index));
                printf("x:%d gsrc_dst_value:%d_%d_%f\n", x / 3, g_index, g_offset_index, *(psrc + g_offset_index));
                printf("x:%d bsrc_dst_value:%d_%d_%f\n", x / 3, b_index, b_offset_index, *(psrc + b_offset_index));
                fflush(stdout);
            }
            *(pdst + r_index) = *(psrc + r_offset_index);
            *(pdst + g_index) = *(psrc + g_offset_index);
            *(pdst + b_index) = *(psrc + b_offset_index);
        }
    }
    pdst += width * height * 3;
}

inline void flip(float *&psrc, float *&pdst, int tuple_w, int tuple_h, int debug) {
    for (int y = 0; y < tuple_h; ++y) {
        for (int x = 0; x < tuple_w * 3 / 2 + 1; x += 3) {
            int rline_index = x;
            int gline_index = x + 1;
            int bline_index = x + 2;
            int rows = tuple_w * 3;
            int r_mirror = rows - rline_index - 1 - 2;
            int g_mirror = rows - rline_index - 1 - 1;
            int b_mirror = rows - rline_index - 1;
            int pre_r_index = y * tuple_w + rline_index;
            int pre_g_index = y * tuple_w + gline_index;
            int pre_b_index = y * tuple_w + bline_index;
            int last_r_index = y * tuple_w + r_mirror;
            int last_g_index = y * tuple_w + g_mirror;
            int last_b_index = y * tuple_w + b_mirror;
            if (y == 0 && debug) {
                printf("x:%d rpre_last_value:%d_%d_%f_%f\n", x / 3, pre_r_index, last_r_index, *(psrc + pre_r_index),
                       *(psrc + last_r_index));
                printf("x:%d gpre_last_value:%d_%d_%f_%f\n", x / 3, pre_g_index, last_g_index, *(psrc + pre_g_index),
                       *(psrc + last_g_index));
                printf("x:%d bpre_last_value:%d_%d_%f_%f\n", x / 3, pre_b_index, last_b_index, *(psrc + pre_b_index),
                       *(psrc + last_b_index));
                fflush(stdout);
            }

            *(pdst + pre_r_index) = *(psrc + last_r_index);
            *(pdst + last_r_index) = *(psrc + pre_r_index);
            *(pdst + pre_g_index) = *(psrc + last_g_index);
            *(pdst + last_g_index) = *(psrc + pre_g_index);
            *(pdst + pre_b_index) = *(psrc + last_b_index);
            *(pdst + last_b_index) = *(psrc + pre_b_index);
        }
    }
    psrc += tuple_w * tuple_h * 3;
    pdst += tuple_w * tuple_h * 3;
}

void ex_pic(float *phead) {
    float *tmp = phead;
    slog::info << "Star to ex_pic" << slog::endl;

    /** 图片路径 **/
    const char *img_dir = FLAGS_i.c_str();
    /** 读取图片 **/
    cv::Mat image = cv::imread(img_dir);

    cv::Mat resized;
    /** 图片大小转换 **/
    cv::resize(image, resized, cv::Size(256, 256));

    /** 读取均值化文件 **/
    slog::info << "Star to load mean.bin file" << slog::endl;
    FILE *pInputFile = fopen("/home/topn-demo/C2319_Mean.binimg", "rb");
    int width, height, channel;
    size_t read_num;
    read_num = fread(&width, 4, 1, pInputFile);
    read_num = fread(&height, 4, 1, pInputFile);
    read_num = fread(&channel, 4, 1, pInputFile);
    slog::info << "load mean.bin #" << width << "_" << height << "_" << channel << slog::endl;

    float mean_arr[width * height * channel];
    read_num = fread((void *) mean_arr, sizeof(float), (size_t) width * height * channel, pInputFile);
    fclose(pInputFile);
    if (width * height * channel != resized.rows * resized.cols * resized.channels()) {
        slog::info << "dim error ! the mean file data length is not equal image size" << slog::endl;
        throw std::logic_error("dim error ! the mean file data length is not equal image size");
    }

    float d_mean[width * height * channel];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float rs = sub_mean(resized, mean_arr, x, y, width, 2, 0);
            float gs = sub_mean(resized, mean_arr, x, y, width, 1, 1);
            float bs = sub_mean(resized, mean_arr, x, y, width, 0, 2);
            d_mean[y * width + x] = rs;
            d_mean[y * width + x + width * height] = gs;
            d_mean[y * width + x + width * height * 2] = bs;
        }
    }
    resized.release();

    crop(d_mean, phead, 0, 0, 224, 224, 0);
    crop(d_mean, phead, 0, 11, 224, 224, 0);
    crop(d_mean, phead, 32, 21, 224, 224, 0);
    crop(d_mean, phead, 32, 32, 224, 224, 0);
    flip(tmp, phead, 224, 224, 0);
    flip(tmp, phead, 224, 224, 0);
    flip(tmp, phead, 224, 224, 0);
    flip(tmp, phead, 224, 224, 0);
}

void fillData(InferRequest &inferRequest, CNNNetReader &reader) {

    InputsDataMap inputInfo = reader.getNetwork().getInputsInfo();
    for (const auto &item : inputInfo) {
        Blob::Ptr input = inferRequest.GetBlob(item.first);

        FILE *pInputFile = fopen("/home/topn-demo/test_input.bin", "rb");
        float pInput[8 * 224 * 224 * 3];
        float pInput2[224 * 224 * 3];
        size_t read = fread((void *) pInput2, sizeof(float), (size_t) 224 * 224 * 3, pInputFile);
//        read = fread((void *) pInput2, sizeof(float), (size_t) 224 * 224 * 3, pInputFile);
        ex_pic(pInput);
        float sum = 0;
        int offset = 224 * 224 * 3 * 0;
        for (int j = 0; j < 224 * 224 * 3; ++j) {
            sum += abs(pInput2[j] - pInput[offset + j]);
            if (j % 10 == 0) {
                printf("sum %f \n", sum);
            }
        }
        printf("diff %f \n", sum / (224 * 224 * 3));
        exit(0);
//        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
//
//        for (size_t i = 0; i < (8 * 224 * 224 * 3); ++i) {
//            data[i] = pInput[i];
//        }
//        fclose(pInputFile);
    }
}

const int NET_SIZE = 1;

/**
 * 背景:在有两个物理核的服务器上,使用MKLDNNPlugin插件加载一个网络,在推断过程中发现只使用了一个物理核,另一个物理核处于空闲状态。
 * 参考官方文档中的:It creates an executable network from a network object. The executable network is associated with single hardware device. It's possible to create as many networks as needed and to use them simultaneously (up to the limitation of the hardware resources)
 * 所以使用同一个插件加载第二个网络,但是在推断第二个网络的过程中发现第二个网络会使用所有的cpu资源,而第一个网络仍然像之前说的只使用一个物理核
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {
    slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

    /** 参数转换/验证 */
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    ExecutableNetwork executableNetwork[NET_SIZE];
    InferRequest inferRequest[NET_SIZE];
    InferencePlugin plugin[NET_SIZE];
    CNNNetReader reader[NET_SIZE];


    createPlugin(plugin[0]);
    readNet(reader[0]);

    for (int i = 0; i < NET_SIZE; i++) {
        executableNetwork[i] = plugin[0].LoadNetwork(reader[0].getNetwork(), {});
        inferRequest[i] = executableNetwork[i].CreateInferRequest();
        fillData(inferRequest[i], reader[0]);
    }

    pthread_t callThd[NET_SIZE];
    for (int i = 0; i < NET_SIZE; i++) {
        int rc = pthread_create(&callThd[i], NULL, run, (void *) &inferRequest[NET_SIZE - 1 - i]);
    }
    for (int i = 0; i < NET_SIZE; i++) {
        pthread_join(callThd[i], NULL);
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}