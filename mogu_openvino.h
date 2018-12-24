//
// Created by Tom Wang on 2018/12/23.
//

#ifndef DDUP_MOGU_OPENVINO_H
#define DDUP_MOGU_OPENVINO_H

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
/**
 * 输入层图片信息(必须)
 */
struct ImageInfo {
    /**
     * 图片三维
     */
    int width, height, channel;
    /**
     * 翻转信息
     */
    int flip;
    /**
     * 均值文件
     */
    std::string meanFile;
    /**
     * 归一化数值
     */
    float scale;
    /**
     * 图片剪裁
     */
    int corpSize_W, cropSize_H, cropNum;
    /**
     * 剪裁左上角点坐标
     */
    int corpPoint[][2];
};

struct Config {
    // ----------------------------------------必须参数--------------------------------//

    /**
     * 模型存放路径,模型名
     */
    std::string modelDir, modelName;
    /**
     * 网络输入图片信息
     */
    ImageInfo *pImageInfo;

    // ----------------------------------------必须但默认参数---------------------------//

    /**
     * 驱动设备
     * 关系到推断引擎加载相对应的plugin
     */
    InferenceEngine::TargetDevice targetDevice = InferenceEngine::TargetDevice::eCPU;
    /**
     * 输入层图片布局
     */
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NHWC;
    /**
     * 输出层图片布局
     */
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NC;
    /**
     * 输入层数据精度
     */
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    /**
     * 输出层数据精度
     */
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    void toString() {
        printf("Config information:\n"
               "modelDir:%s\n"
               "modelName:%s\n"
               "width_height_channel:%d_%d_%d\n"
               "flip:%d\n"
               "meanFile:%s\n"
               "scale:%f\n"
               "corpW_cropH_cropN:%d_%d_%d\n", modelDir.c_str(), modelName.c_str(), pImageInfo->width,
               pImageInfo->height, pImageInfo->channel, pImageInfo->flip, pImageInfo->meanFile.c_str(),
               pImageInfo->scale,
               pImageInfo->corpSize_W, pImageInfo->cropSize_H, pImageInfo->cropNum);
    }
};

struct Output {

    float *data;
    /**
     * 输出shape
     */
    int shape[];
};

class Openvino_Net {
public:
    Openvino_Net(Config &config) : config(config), pPlugin(NULL), pReader(NULL), meanArr(NULL) {}

    /**
    * 构建一个openvino的推断引擎
    */
    int create_inf_engine();

    /**
     * 推断
     */
    Output *inference(unsigned char *pImageHead, int imageW, int imageH);

private:
    /**
     * 可执行网络结构
     */
    ExecutableNetwork executableNetwork;
    /**
     * 插件
     */
    InferencePlugin *pPlugin;
    /**
     * 模型网络信息
     */
    CNNNetReader *pReader;
    /**
     * 配置信息
     */
    Config config;
    /**
     * 均值数组头指针
     */
    float *meanArr;

    /**
    * 读取配置文件
    */
    int read_config();
    /**
    * 读取模型网络信息
    */
    int read_net();
    /**
    * 填充请求数据
    */
    void fill_data(InferRequest &inferRequest, Config &config, unsigned char *pImageHead, int imageW, int imageH);
    /**
    * 收集推断结果
    */
    void collectOutPut(InferRequest &inferRequest, Config &config, Output &output);
    /**
    * 图片增强逻辑
    */
    void ex_pic(float *phead, Config &config, unsigned char *pImageHead, int imageW, int imageH);
};

#endif //DDUP_MOGU_OPENVINO_H
