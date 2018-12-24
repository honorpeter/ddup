//
// Created by Tom Wang on 2018/12/23.
//

#include "mogu_openvino.h"


static std::map<std::string, ExecutableNetwork *> execNetMap;
static std::map<std::string, Config *> configMap;
static std::map<std::string, float *> meanMap;
static std::map<std::string, CNNNetReader *> readerMap;

/**
 * 检查配置信息是否完整
 * @param config  配置信息
 * @return 检查结果
 */
inline int assertConfig(Config &config) {
    /** 检查模型文件路径 **/
    if (config.modelDir.empty()) {
        return 0;
    }

    /** 检查模型名称**/
    if (config.modelName.empty()) {
        return 0;
    }
    /** 检查图片**/
    if (!config.pImageInfo) {
        return 0;
    }
    return 1;
}

/**
 * 数组对应下标相减
 */
inline float
sub_mean(cv::Mat &image, const float *mean_arr, int x, int y, int width, float &scale, int c, int mean_delta_a) {
    unsigned char r = image.at<cv::Vec3b>(y, x)[c];
    float mean_r = mean_arr[y * width + x + mean_delta_a * width * width];
    return (r - mean_r) / scale;
}

/**
 * 剪裁图片
 */
inline void
crop(const float *psrc, float *&pdst, int &x_offset, int &y_offset, int &width, int &height, int &originW, int &originH) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width * 3; x += 3) {
            int r_index = y * width * 3 + x;
            int g_index = r_index + 1;
            int b_index = r_index + 2;
            int r_offset_index = (y + y_offset) * originW + x / 3 + x_offset;
            int g_offset_index = (y + y_offset) * originW + x / 3 + x_offset + originW * originH;
            int b_offset_index = (y + y_offset) * originW + x / 3 + x_offset + originW * originW * 2;
            *(pdst + r_index) = *(psrc + r_offset_index);
            *(pdst + g_index) = *(psrc + g_offset_index);
            *(pdst + b_index) = *(psrc + b_offset_index);
        }
    }
    pdst += width * height * 3;
}

/**
 * 图片翻转
 */
inline void flip(float *&psrc, float *&pdst, int tuple_w, int tuple_h) {
    for (int y = 0; y < tuple_h; ++y) {
        for (int x = 0; x < tuple_w * 3 / 2; x += 3) {
            int rline_index = x;
            int gline_index = x + 1;
            int bline_index = x + 2;
            int rows = tuple_w * 3;
            int r_mirror = rows - rline_index - 1 - 2;
            int g_mirror = rows - rline_index - 1 - 1;
            int b_mirror = rows - rline_index - 1;
            int pre_r_index = y * tuple_w * 3 + rline_index;
            int pre_g_index = y * tuple_w * 3 + gline_index;
            int pre_b_index = y * tuple_w * 3 + bline_index;
            int last_r_index = y * tuple_w * 3 + r_mirror;
            int last_g_index = y * tuple_w * 3 + g_mirror;
            int last_b_index = y * tuple_w * 3 + b_mirror;
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

/**
 * 构建插件
 * @param plugin 插件
 */
inline void create_plugin(InferencePlugin &plugin,Config &config) {
    // todo 绝对路径
    InferenceEnginePluginPtr engine_ptr = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getSuitablePlugin(
            config.targetDevice);
    plugin = InferencePlugin(engine_ptr);
    plugin.SetConfig({{PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES}});
}

/**
 * 读取配置文件
 * @param config 配置
 * @return
 */
int read_config(Config &config) {
    char configDir[config.modelDir.size() + config.modelName.size() + 8];
    sprintf(configDir, "%s/%s.config", config.modelDir.c_str(), config.modelName.c_str());

    FILE *pConfigFile = fopen(configDir, "r");
    if (pConfigFile == NULL) {
        return 1;
    }

    int readNum = 0;
    /** 读取图片三维 **/
    int width, height, channel;
    readNum = fscanf(pConfigFile, "width_height_channel=%d_%d_%d\n", &width, &height, &channel);
    if (!readNum) {
        return 0;
    }
    if (width > 0 && height > 0 && channel > 0) {
        config.pImageInfo->width = width;
        config.pImageInfo->height = height;
        config.pImageInfo->height = height;
    }

    /** 读取图片翻转信息 **/
    readNum =fscanf(pConfigFile, "flip=%d\n", &config.pImageInfo->flip);

    /** 读取均值文件 **/
    // todo doudi
    char buffer[512];
    int meanSize;
    int meanH, meanW, meanC;
    readNum = fscanf(pConfigFile, "meanFile=%s\n", buffer);
    std::string meanFileStr(buffer);
    if (!meanFileStr.empty()) {
        config.pImageInfo->meanFile = meanFileStr;
        FILE *pMeanFile = fopen(buffer, "rb");

        readNum = fread(&meanH, 4, 1, pMeanFile);
        readNum = fread(&meanW, 4, 1, pMeanFile);
        readNum = fread(&meanC, 4, 1, pMeanFile);
        meanSize = width * height * channel;

        //todo malloc
        float *tmpMean = (float *) malloc(sizeof(float) * meanSize);
        if (fread((void *) tmpMean, sizeof(float), (size_t)meanSize, pMeanFile) != meanSize) {
            return 0;
        }
        meanMap.insert(std::map<std::string, float *>::value_type(config.modelName, tmpMean));
    }

    /** 读取归一化系数 **/
    readNum =fscanf(pConfigFile, "scale=%f\n", &config.pImageInfo->scale);

    /** 读取裁剪大小和数目 **/
    readNum =fscanf(pConfigFile, "corpW_cropH_cropN=%d_%d_%d\n", &config.pImageInfo->corpSize_W, &config.pImageInfo->cropSize_H,
           &config.pImageInfo->cropNum);

    /** 读取裁剪的起始点 **/
    int xPoint, yPoint;
    for (int i = 0; i < config.pImageInfo->cropNum; ++i) {
        readNum =fscanf(pConfigFile, "x_y=%d_%d\n", &xPoint, &yPoint);
        config.pImageInfo->corpPoint[i][0] = xPoint;
        config.pImageInfo->corpPoint[i][1] = yPoint;
    }

    fclose(pConfigFile);
    return 0;
}

/**
 * 读取模型网络信息
 * @param networkReader 网络信息
 * @param config  配置
 * @return 成功与否
 *
 */
int read_net(CNNNetReader &networkReader, Config &config) {
    /** 格式化模型描述文件路径,权重路径 **/
    char xmlDir[config.modelDir.size() + config.modelName.size() + 7];
    char binDir[config.modelDir.size() + config.modelName.size() + 7];

    sprintf(xmlDir, "%s/%s.xml", config.modelDir.c_str(), config.modelName.c_str());
    sprintf(binDir, "%s/%s.bin", config.modelDir.c_str(), config.modelName.c_str());

    std::string xmlDirStr(xmlDir);
    std::string binDirStr(binDir);

    /** 读取模型文件 **/
    networkReader.ReadNetwork(xmlDirStr);
    networkReader.ReadWeights(binDirStr);
    CNNNetwork network = networkReader.getNetwork();

    /** 设置输入精度和布局 **/
    InputsDataMap inputInfo = network.getInputsInfo();
    for (auto &inputInfoItem : inputInfo) {
        InputInfo::Ptr inputData = inputInfoItem.second;
        if (!inputData) {
            return 0;
        }
        inputData->setPrecision(config.inputPrecision);
        inputData->setLayout(config.inputLayout);
    }

    /** 计算batch_size 大小 **/
    size_t batchSize = 1;
    int cropNum = config.pImageInfo->cropNum;
    batchSize = cropNum == 0 ? batchSize : cropNum;
    batchSize = config.pImageInfo->flip == 1 ? 2 * batchSize : batchSize;
    network.setBatchSize(batchSize);

    /** 设置输出精度和布局 **/
    OutputsDataMap outputInfo(network.getOutputsInfo());
    for (auto &item : outputInfo) {
        DataPtr outputData = item.second;
        if (!outputData) {
            return 0;
        }
        outputData->setPrecision(config.outputPrecision);
        outputData->setLayout(config.outputLayout);
    }
    return 1;
}

/**
 * 图片增强逻辑
 */
void ex_pic(float *phead, Config &config, unsigned char *pImageHead, int imageW, int imageH) {
    float *tmp = phead;

    int width = config.pImageInfo->height;
    int height = config.pImageInfo->width;
    int channel = config.pImageInfo->channel;
    int targetW = width - config.pImageInfo->corpSize_W;
    int targetH = height - config.pImageInfo->cropSize_H;
    int cropNum = config.pImageInfo->cropNum;
    float d_mean[width * height * channel];

    // todo RGB OR BGR 判断逻辑

    /** 读取图片 **/
    cv::Mat image(imageH, imageW, CV_8UC3, pImageHead);

    /** 图片大小转换 **/
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height));

    /** 从资源池读取均值数组 **/
    auto meanIterator = meanMap.find(config.modelName);
    float *meanArr = meanIterator->second;
    if (meanArr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float rs = sub_mean(resized, meanArr, x, y, width,config.pImageInfo->scale, 2, 0);
                float gs = sub_mean(resized, meanArr, x, y, width,config.pImageInfo->scale, 1, 1);
                float bs = sub_mean(resized, meanArr, x, y, width,config.pImageInfo->scale, 0, 2);
                d_mean[y * width + x] = rs;
                d_mean[y * width + x + width * height] = gs;
                d_mean[y * width + x + width * height * 2] = bs;
            }
        }
    } else {
        for (int y = 0; y<resized.rows; ++y){
            for (int x = 0; x < resized.cols; ++x) {
                d_mean[y * resized.cols + x] = resized.at<cv::Vec3b>(y, x)[2] * 1.0f;
                d_mean[y * width + x + width * height] = resized.at<cv::Vec3b>(y, x)[1] * 1.0f;
                d_mean[y * width + x + width * height * 2] = resized.at<cv::Vec3b>(y, x)[0] * 1.0f;
            }
        }
    }

    /** 释放大小转换的中间数据 **/
    resized.release();

    /** 裁剪逻辑 **/
    if (cropNum > 0) {
        for (int i = 0; i < cropNum; ++i) {
            crop(d_mean, phead, config.pImageInfo->corpPoint[i][0], config.pImageInfo->corpPoint[i][1], targetW,
                 targetH, width, height);
        }
    }

    /** 翻转逻辑 **/
    if (config.pImageInfo->flip) {
        for (int i = 0; i < cropNum; ++i) {
            flip(tmp, phead, targetW, targetH);
        }
    }
}

/**
 * 填充请求数据
 */
void fill_data(InferRequest &inferRequest, Config &config, unsigned char *pImageHead, int imageW, int imageH) {
    InputsDataMap inputInfo;

    /** 从资源池获取网络信息 **/
    auto readerIterator = readerMap.find(config.modelName);
    inputInfo = readerIterator->second->getNetwork().getInputsInfo();

    /** 遍历输入层信息,进行数据填充 **/
    for (const auto &item : inputInfo) {
        Blob::Ptr input = inferRequest.GetBlob(item.first);
        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        ex_pic(data, config, pImageHead, imageW, imageH);
    }
}

/**
 * 收集推断结果
 */
void collectOutPut(InferRequest &inferRequest, Config &config, Output &output) {
    OutputsDataMap outputInfo;

    /** 从资源池获取网络信息 **/
    auto readerIterator = readerMap.find(config.modelName);
    outputInfo = readerIterator->second->getNetwork().getOutputsInfo();

    /** 遍历输出层信息,进行结果填充 **/
    for (const auto &item : outputInfo) {
        Blob::Ptr outputBlob = inferRequest.GetBlob(item.first);
        const LockedMemory<const void> memLocker = outputBlob->cbuffer();
        const float *outputBuffer = memLocker.as<PrecisionTrait<Precision::FP32>::value_type *>();
        size_t batchSize = outputBlob->getTensorDesc().getDims()[0];
        size_t dim = outputBlob->getTensorDesc().getDims()[1];
    }
}

/**
 * 构建一个openvino的推断引擎
 */
int create_inf_engine(Config &config) {

    /** 参数检查 **/
    if (!assertConfig(config)) {
        return 0;
    }

    // 生命周期验证
    InferencePlugin plugin;
    CNNNetReader reader;
    ExecutableNetwork executableNetwork;

    /** 初始化插件 **/
    create_plugin(plugin, config);
    /** 读取配置文件,填充/覆盖 缺省配置 **/
    read_config(config);
    config.toString(); // debug逻辑
    /** 读取模型网络信息 **/
    read_net(reader, config);
    /** 插件通过网络信息加载称可执行网络 **/
    executableNetwork = plugin.LoadNetwork(reader.getNetwork(), {});
    /** 将可执行网络注册至资源池 **/
    execNetMap.insert(std::map<std::string, ExecutableNetwork *>::value_type(config.modelName, &executableNetwork));
    /** 将配置信息注册至资源池 **/
    configMap.insert(std::map<std::string, Config *>::value_type(config.modelName, &config));
    /** 将网络信息注册至资源池 **/
    readerMap.insert(std::map<std::string, CNNNetReader *>::value_type(config.modelName, &reader));

    return 1;
}

/**
 * 推断
 */
Output *inference(std::string &modelName, unsigned char *pImageHead, int imageW, int imageH) {
    if (modelName.empty()) {
        return NULL;
    }

    printf("Star to Inference\n"); // debug逻辑

    ExecutableNetwork *pExecutableNetwork;
    Config *pConfig;
    Output *output = NULL;

    /** 从资源池中寻找可执行网络和配置信息 **/
    auto execIterator = execNetMap.find(modelName);
    printf("Get execNet %s\n", execIterator->first.c_str());
    pExecutableNetwork = execIterator->second;
    auto configIteator = configMap.find(modelName);
    printf("Get config %s\n", configIteator->first.c_str());
    pConfig = configIteator->second;

    printf("End to find...");

    /** 创建请求 **/
    InferRequest inferRequest = pExecutableNetwork->CreateInferRequest();
    printf("Star to fill_data\n"); // debug逻辑
    /** 填充请求数据 **/
    fill_data(inferRequest, *pConfig, pImageHead, imageW, imageH);
    /** 进行推断 **/
    inferRequest.Infer();
    /** 收集输出层结果 **/
    collectOutPut(inferRequest, *pConfig, *output);
    return output;
}

// --------------------------------------------------测试函数区-------------------------------------------------//

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

int main(int argc, char *argv[]){
    slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

    /** 参数转换/验证 */
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    Config config;
    config.modelDir = FLAGS_m;
    config.modelName = std::string("dl_model_tmp");
    ImageInfo imageInfo;
    config.pImageInfo = &imageInfo;
    create_inf_engine(config);

    /** 图片路径 **/
    const char *img_dir = FLAGS_i.c_str();
    /** 读取图片 **/
    cv::Mat image = cv::imread(img_dir);
    unsigned char imageArr[256][256][3];
    Output *output = inference(config.modelName, &imageArr[0][0][0], 256, 256);
}