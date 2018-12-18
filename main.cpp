/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <unistd.h>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <pthread.h>

#include "classification_sample.h"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

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

void *run(void *p){

    InferRequest *infer_request = (InferRequest *)p;

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


/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/
int main(int argc, char *argv[]) {
//    try {
//        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
//
//        // ------------------------------转换/验证输入参数 ---------------------------------
//        if (!ParseAndCheckCommandLine(argc, argv)) {
//            return 0;
//        }
//
//        /** 获取输入文件 **/
//        std::vector<std::string> imageNames;
//        parseInputFilesArguments(imageNames);
//        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
//        slog::info << "Loading plugin" << slog::endl;
//
//        /* 插件 1 */
//        InferencePlugin plugin = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getPluginByDevice(FLAGS_d);
//
//        if (FLAGS_p_msg) {
//            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
//        }
//
//        /** 加载默认插件 **/
//        if (FLAGS_d.find("CPU") != std::string::npos) {
//            /**
//             * cpu_extensions library is compiled from "extension" folder containing
//             * custom MKLDNNPlugin layer implementations. These layers are not supported
//             * by mkldnn, but they can be useful for inferring custom topologies.
//            **/
//            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
//        }
//
//        if (!FLAGS_l.empty()) {
//            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
//            auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
//            plugin.AddExtension(extension_ptr);
//            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
//        }
//        if (!FLAGS_c.empty()) {
//            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
//            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
//            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
//        }
//
//        /** Setting plugin parameter for collecting per layer metrics **/
//        if (FLAGS_pc) {
//            plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
//        }
//
//        plugin.SetConfig({{PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES}});
//        /** 打印插件版本 **/
//        printPluginVersion(plugin, std::cout);
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
//        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
//        slog::info << "Loading network files:"
//                      "\n\t" << FLAGS_m <<
//                   "\n\t" << binFileName <<
//                   slog::endl;
//
//        CNNNetReader networkReader;
//        /** 读取模型描述文件 **/
//        networkReader.ReadNetwork(FLAGS_m);
//
//        /** 读取模型权重文件 **/
//        networkReader.ReadWeights(binFileName);
//        CNNNetwork network = networkReader.getNetwork();
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 3. Configure input & output ---------------------------------------------
//
//        // --------------------------- Prepare input blobs -----------------------------------------------------
//        slog::info << "Preparing input blobs" << slog::endl;
//
//        /**获取所有的输入*/
//        InputsDataMap inputInfo = network.getInputsInfo();
//        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
//        /** 获取第一个输入*/
//        auto inputInfoItem = *inputInfo.begin();
//
//        /** Specifying the precision and layout of input data provided by the user.
//         * This should be called before load of the network to the plugin **/
//        inputInfoItem.second->setPrecision(Precision::FP32);
//        inputInfoItem.second->setLayout(Layout::NHWC);
//
//        /** Setting batch size using image count **/
//        network.setBatchSize(imageNames.size());
//        size_t batchSize = network.getBatchSize();
//        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
//
//        // ------------------------------ Prepare output blobs -------------------------------------------------
//        slog::info << "Preparing output blobs" << slog::endl;
//
//        OutputsDataMap outputInfo(network.getOutputsInfo());
//        // BlobMap outputBlobs;
//        std::string firstOutputName;
//
//        for (auto &item : outputInfo) {
//            if (firstOutputName.empty()) {
//                firstOutputName = item.first;
//            }
//            DataPtr outputData = item.second;
//            if (!outputData) {
//                throw std::logic_error("output data pointer is not valid");
//            }
//            outputData->setPrecision(Precision::FP32);
//            outputData->setLayout(Layout::NCHW);
//        }
//
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 4. Loading model to the plugin ------------------------------------------
//        slog::info << "Loading model to the plugin" << slog::endl;
//
//        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
//        inputInfoItem.second = {};
//        outputInfo = {};
//        network = {};
//        networkReader = {};
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 5. Create infer request -------------------------------------------------
//        InferRequest infer_request = executable_network.CreateInferRequest();
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 6. Prepare input --------------------------------------------------------
//
//
//        /** 遍历所有输入 **/
//        for (const auto &item : inputInfo) {
//            /** Creating input blob **/
//            Blob::Ptr input = infer_request.GetBlob(item.first);
//
//            /** Filling input tensor with images. First b channel, then g and r channels **/
//            size_t num_channels = input->getTensorDesc().getDims()[1];
//            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];
//
//            FILE *pInputFile = fopen(imageNames[0].c_str(), "rb");
//            float pInput[num_channels * image_size];
//            size_t readed_num = fread((void *) pInput, sizeof(float), num_channels * image_size, pInputFile);
//
//            slog::info << "Loading input" << imageNames[0] << " to the data" << slog::endl;
//            slog::info << "input size " << readed_num << " float" << slog::endl;
//
//            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
//
//            /** Iterate over all input images **/
//            for (size_t i = 0; i < (num_channels * image_size); ++i) {
//                data[i] = pInput[i];
//            }
//        }
//        inputInfo = {};
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 7. Do inference ---------------------------------------------------------
//        slog::info << "Starting inference (" << FLAGS_ni << " iterations)" << slog::endl;
//
//        typedef std::chrono::high_resolution_clock Time;
//        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
//        typedef std::chrono::duration<float> fsec;
//
//        double total = 0.0;
//        /** Start inference & calc performance **/
//        for (int iter = 0; iter < FLAGS_ni; ++iter) {
//            auto t0 = Time::now();
//            infer_request.Infer();
//            auto t1 = Time::now();
//            fsec fs = t1 - t0;
//            ms d = std::chrono::duration_cast<ms>(fs);
//            total += d.count();
//        }
//        // -----------------------------------------------------------------------------------------------------
//
//        // --------------------------- 8. Process output -------------------------------------------------------
//        slog::info << "Processing output blobs" << slog::endl;
//
//        const Blob::Ptr output_blob = infer_request.GetBlob(firstOutputName);
//
//        /** Validating -nt value **/
//        const int resultsCnt = output_blob->size() / batchSize;
//        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
//            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
// << resultsCnt + 1 << " and more than 0)\n            will be used maximal value : " << resultsCnt;
//            FLAGS_nt = resultsCnt;
//        }
//
//        /** This vector stores id's of top N results **/
//        std::vector<unsigned> results;
//        std::cout << std::endl << "Top " << FLAGS_nt << " results:" << std::endl << std::endl;
//
//        const LockedMemory<const void> memLocker = output_blob->cbuffer();
//        const float *output_buffer = memLocker.as<PrecisionTrait<Precision::FP32>::value_type *>();
//
//        slog::info << "Print fea \n" << slog::endl;
//
//        for (int i = 0; i < 512; i++) {
//            float fea = *(output_buffer + i);
//            std::cout << std::endl << "_" << fea << std::endl;
//        }
//
//        // -----------------------------------------------------------------------------------------------------
//        std::cout << std::endl << "total inference time: " << total << std::endl;
//        std::cout << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms"
//                  << std::endl;
//        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS"
//                  << std::endl;
//        std::cout << std::endl;
//
//        /** Show performance results **/
//        if (FLAGS_pc) {
//            printPerformanceCounts(infer_request, std::cout);
//        }
//    }
//    catch (const std::exception &error) {
//        slog::err << "" << error.what() << slog::endl;
//        return 1;
//    }
//    catch (...) {
//        slog::err << "Unknown/internal exception happened." << slog::endl;
//        return 1;
//    }

    slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

    // ------------------------------转换/验证输入参数 ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    /** 插件1 **/
    InferencePlugin plugin = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getPluginByDevice(FLAGS_d);

    /** 加载cpu插件 **/
    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
         * cpu_extensions library is compiled from "extension" folder containing
         * custom MKLDNNPlugin layer implementations. These layers are not supported
         * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (!FLAGS_l.empty()) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
        plugin.AddExtension(extension_ptr);
        slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
    }
    if (!FLAGS_c.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
    }

    /** Setting plugin parameter for collecting per layer metrics **/
    if (FLAGS_pc) {
        plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
    }
    printPluginVersion(plugin, std::cout);

    /** 插件2 **/
    InferencePlugin plugin2 = PluginDispatcher({FLAGS_pp, "../../../lib/intel64", ""}).getPluginByDevice(FLAGS_d);

    /** 加载cpu插件 **/
    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
         * cpu_extensions library is compiled from "extension" folder containing
         * custom MKLDNNPlugin layer implementations. These layers are not supported
         * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin2.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (!FLAGS_l.empty()) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
        plugin2.AddExtension(extension_ptr);
        slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
    }
    if (!FLAGS_c.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin2.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
    }

    /** Setting plugin parameter for collecting per layer metrics **/
    if (FLAGS_pc) {
        plugin2.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
    }
    printPluginVersion(plugin2, std::cout);

    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    slog::info << "Loading network files:"
                  "\n\t" << FLAGS_m <<
               "\n\t" << binFileName <<
               slog::endl;

    CNNNetReader networkReader;
    /** 读取模型描述文件 **/
    networkReader.ReadNetwork(FLAGS_m);

    /** 读取模型权重文件 **/
    networkReader.ReadWeights(binFileName);
    CNNNetwork network = networkReader.getNetwork();

    slog::info << "Preparing input blobs" << slog::endl;

    /**获取所有的输入*/
    InputsDataMap inputInfo = network.getInputsInfo();
    if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
    /** 获取第一个输入*/
    auto inputInfoItem = *inputInfo.begin();

    /** Specifying the precision and layout of input data provided by the user.
     * This should be called before load of the network to the plugin **/



    /** Setting batch size using image count **/
    network.setBatchSize(8);
    size_t batchSize = network.getBatchSize();
    slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

    // ------------------------------ Prepare output blobs -------------------------------------------------
    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputInfo(network.getOutputsInfo());
    // BlobMap outputBlobs;
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


    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Loading model to the plugin ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    inputInfoItem.second = {};
    outputInfo = {};
    network = {};
    networkReader = {};

    InferRequest infer_request = executable_network.CreateInferRequest();
    /** 遍历所有输入 **/
    for (const auto &item : inputInfo) {
        /** Creating input blob **/
        Blob::Ptr input = infer_request.GetBlob(item.first);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_channels = input->getTensorDesc().getDims()[1];
        size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

        FILE *pInputFile = fopen("/home/topn-demo/test_input.bin", "rb");
        float pInput[8*224*224*3];
        size_t readed_num = fread((void *) pInput, sizeof(float), 8*224*224*3, pInputFile);

        slog::info << "input size " << readed_num << " float" << slog::endl;

        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

        /** Iterate over all input images **/
        for (size_t i = 0; i < (8*224*224*3); ++i) {
            data[i] = pInput[i];
        }

        fclose(pInputFile);
    }
    inputInfo = {};


    //  ------------- 网络2 ---------------
    CNNNetReader networkReader2;
    /** 读取模型描述文件 **/
    networkReader2.ReadNetwork(FLAGS_m);

    /** 读取模型权重文件 **/
    networkReader2.ReadWeights(binFileName);
    CNNNetwork network2 = networkReader2.getNetwork();

    slog::info << "Preparing input blobs" << slog::endl;

    /**获取所有的输入*/
    InputsDataMap inputInfo2 = network2.getInputsInfo();
    if (inputInfo2.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
    /** 获取第一个输入*/
    auto inputInfoItem2 = *inputInfo2.begin();

    /** Specifying the precision and layout of input data provided by the user.
     * This should be called before load of the network to the plugin **/
    inputInfoItem2.second->setPrecision(Precision::FP32);
    inputInfoItem2.second->setLayout(Layout::NHWC);

    /** Setting batch size using image count **/
    network2.setBatchSize(8);

    // ------------------------------ Prepare output blobs -------------------------------------------------
    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputInfo2(network2.getOutputsInfo());
    // BlobMap outputBlobs;
    std::string firstOutputName2;

    for (auto &item : outputInfo2) {
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

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Loading model to the plugin ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network2 = plugin.LoadNetwork(network2, {});
    inputInfoItem2.second = {};
    outputInfo2 = {};
    network2 = {};
    networkReader2 = {};

    InferRequest infer_request2 = executable_network2.CreateInferRequest();
    /** 遍历所有输入 **/
    for (const auto &item : inputInfo2) {
        /** Creating input blob **/
        Blob::Ptr input = infer_request2.GetBlob(item.first);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_channels = input->getTensorDesc().getDims()[1];
        size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

        FILE *pInputFile = fopen("/home/topn-demo/test_input.bin", "rb");
        float pInput[8*224*224*3];
        size_t readed_num = fread((void *) pInput, sizeof(float), 8*224*224*3, pInputFile);

        slog::info << "input size " << readed_num << " float" << slog::endl;

        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

        /** Iterate over all input images **/
        for (size_t i = 0; i < (8*224*224*3); ++i) {
            data[i] = pInput[i];
        }
    }
    inputInfo2 = {};

    pthread_t callThd[2];
    int rc = pthread_create(&callThd[0], NULL, run,(void *)&infer_request);
    if (rc){
        printf("ERROR: pthread_create() return %d\n", rc);
        return -1;
    }

    int rc2 = pthread_create(&callThd[1], NULL, run,(void *)&infer_request2);
    if (rc2){
        printf("ERROR: pthread_create() return %d\n", rc);
        return -1;
    }

    slog::info << "Execution successful" << slog::endl;

    pthread_join(callThd[0],NULL);
    pthread_join(callThd[1],NULL);
    return 0;
}
