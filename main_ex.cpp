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
    }
    slog::info << slog::endl;
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

void ex_pic(float *phead, int size) {
    slog::info << "Star to ex_pic" << slog::endl;

    /** 图片路径 **/
    const char *img_dir = FLAGS_i.c_str();
    /** 读取图片 **/
    cv::Mat image = cv::imread(img_dir);

    slog::info << "Star to resize" << slog::endl;

    cv::Mat resized;
    cv::Mat rgb;
    /** 图片大小转换 **/
    cv::resize(image, resized, cv::Size(256, 256));
    /** bgr -> rgb **/
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    size_t mean_data_size = 256 * 256 * 3;
    int delta_green = 256 * 256;
    int delta_blue = 256 * 256 * 2;

    slog::info << "Star to load mean.bin file" << slog::endl;

    float mean_arr[mean_data_size];
    /** 读取均值化文件 **/
    FILE *pInputFile = fopen("/home/topn-demo/C2319_Mean.binimg", "rb");
    int width, height, channel;
    size_t read_num;
    read_num = fread(&width, 4, 1, pInputFile);
    read_num = fread(&height, 4, 1, pInputFile);
    read_num = fread(&channel, 4, 1, pInputFile);
    read_num = fread((void *) mean_arr, sizeof(float), mean_data_size, pInputFile);

    slog::info << "End to load mean.bin file #" << width << "_" << height << "_" << channel << " readNum#" << read_num
               << slog::endl;
    print_head_from_arr(mean_arr, 20);

    if (read_num != rgb.rows * rgb.cols * rgb.channels()) {
        slog::info << "dim error ! the mean file data length is not equal image size" << slog::endl;
        throw std::logic_error("dim error ! the mean file data length is not equal image size");
    }

    /** 均值化 再减去 均值 **/
    for (int i = 0; i < rgb.rows; ++i) {
        for (int z = 0; z < rgb.cols; ++z) {
            rgb.at<cv::Vec3f>(i, z)[0] = (rgb.at<cv::Vec3f>(i, z)[0] - mean_arr[i * z + z]) / 255.0f;
            rgb.at<cv::Vec3f>(i, z)[1] = (rgb.at<cv::Vec3f>(i, z)[1] - mean_arr[i * z + z + delta_green]) / 255.0f;
            rgb.at<cv::Vec3f>(i, z)[2] = (rgb.at<cv::Vec3f>(i, z)[2] - mean_arr[i * z + z + delta_blue]) / 255.0f;
            if (i < 2 && z < 2) {
                print_head_from_arr(&rgb.at<cv::Vec3f>(i, z)[0], 3);
            }
        }
    }

    slog::info << "Star to crop image" << slog::endl;

    /** 图像剪裁 **/
    cv::Mat crop_0_0;
    cv::Mat crop_11_0;
    cv::Mat crop_21_32;
    cv::Mat crop_32_32;
    cv::Rect rect_0_0(0, 0, 224, 224);
    cv::Rect rect_11_0(11, 0, 224, 224);
    cv::Rect rect_21_32(21, 32, 224, 224);
    cv::Rect rect_32_32(32, 32, 224, 224);
    crop_0_0 = rgb(rect_0_0);
    print_head_from_arr(&crop_0_0.at<cv::Vec3f>(0, 0)[0], 3);
    slog::info << "size #" << crop_0_0.rows << "_" << crop_0_0.cols << "_" << crop_0_0.channels() << slog::endl;
    crop_11_0 = rgb(rect_11_0);
    print_head_from_arr(&crop_11_0.at<cv::Vec3f>(0, 0)[0], 3);
    slog::info << "size #" << crop_11_0.rows << "_" << crop_11_0.cols << "_" << crop_11_0.channels() << slog::endl;
    crop_21_32 = rgb(rect_21_32);
    print_head_from_arr(&crop_21_32.at<cv::Vec3f>(0, 0)[0], 3);
    slog::info << "size #" << crop_21_32.rows << "_" << crop_21_32.cols << "_" << crop_21_32.channels() << slog::endl;
    crop_32_32 = rgb(rect_32_32);
    print_head_from_arr(&crop_32_32.at<cv::Vec3f>(0, 0)[0], 3);
    slog::info << "size #" << crop_32_32.rows << "_" << crop_32_32.cols << "_" << crop_32_32.channels() << slog::endl;

    slog::info << "End to crop image" << slog::endl;
    slog::info << "Star to flip image" << slog::endl;

    cv::Mat flip_1;
    cv::Mat flip_2;
    cv::Mat flip_3;
    cv::Mat flip_4;
    cv::flip(crop_0_0, flip_1, 0);
    slog::info << "size #" << flip_1.rows << "_" << flip_1.cols << "_" << flip_1.channels() << slog::endl;
    print_head_from_arr(&flip_1.at<cv::Vec3f>(0, 0)[0], 3);
    cv::flip(crop_11_0, flip_2, 0);
    print_head_from_arr(&flip_2.at<cv::Vec3f>(0, 0)[0], 3);
    cv::flip(crop_21_32, flip_3, 0);
    print_head_from_arr(&flip_3.at<cv::Vec3f>(0, 0)[0], 3);
    cv::flip(crop_32_32, flip_4, 0);
    print_head_from_arr(&flip_4.at<cv::Vec3f>(0, 0)[0], 3);

    slog::info << "End to flip image" << slog::endl;

    if (size < 8 * 224 * 224 * 3) {
        throw std::logic_error("dim error ! the input  data length is not equal batch image size");
    }
    fill_image_2_arr(phead, crop_0_0, 0);
    fill_image_2_arr(phead, crop_11_0, 224 * 224 * 3);
    fill_image_2_arr(phead, crop_21_32, 2 * 224 * 224 * 3);
    fill_image_2_arr(phead, crop_32_32, 3 * 224 * 224 * 3);
    fill_image_2_arr(phead, flip_1, 4 * 224 * 224 * 3);
    fill_image_2_arr(phead, flip_2, 5 * 224 * 224 * 3);
    fill_image_2_arr(phead, flip_3, 6 * 224 * 224 * 3);
    fill_image_2_arr(phead, flip_4, 7 * 224 * 224 * 3);

}

void fillData(InferRequest &inferRequest, CNNNetReader &reader) {

    InputsDataMap inputInfo = reader.getNetwork().getInputsInfo();
    for (const auto &item : inputInfo) {
        Blob::Ptr input = inferRequest.GetBlob(item.first);

        FILE *pInputFile = fopen("/home/topn-demo/test_input.bin", "rb");
        float pInput[8 * 224 * 224 * 3];
        ex_pic(pInput, 8 * 224 * 224 * 3);

        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

        for (size_t i = 0; i < (8 * 224 * 224 * 3); ++i) {
            data[i] = pInput[i];
        }
        fclose(pInputFile);
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