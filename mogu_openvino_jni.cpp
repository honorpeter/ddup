//
// Created by adai on 2018/12/25.
//

#include "mogu_openvino_jni.h"

static std::map<std::string, Openvino_Net *> netPool;

inline void get_string(JNIEnv *env, jstring jstr, std::string &str) {
    const char *pJstr = env->GetStringUTFChars(jstr, nullptr);
    str = std::string(pJstr);
    env->ReleaseStringUTFChars(jstr, pJstr);
}

/*
 * Class:     com_mogujie_algo_openvino_jni_MoguOpenvino
 * Method:    create
 * Signature: (Ljava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_mogujie_algo_openvino_jni_MoguOpenvino_create
        (JNIEnv *env, jclass cls, jstring jModelDir, jstring jModelName){

    std::string modelDirStr;
    std::string modelNameStr;
    get_string(env, jModelDir, modelDirStr);
    get_string(env, jModelName, modelNameStr);
    Config config;
    config.modelDir = modelDirStr;
    config.modelName = modelNameStr;

    auto *net = new Openvino_Net(config);
    net->create_inf_engine();
    netPool.insert(std::map<std::string, Openvino_Net *>::value_type(modelNameStr, net));
    return 1;
}

/*
 * Class:     com_mogujie_algo_openvino_jni_MoguOpenvino
 * Method:    inference
 * Signature: (Ljava/lang/String;[CIII)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_mogujie_algo_openvino_jni_MoguOpenvino_inference
        (JNIEnv * env, jclass cls, jstring mn, jcharArray charArr, jint w, jint h, jint c){

    const char *pJstr = env->GetStringUTFChars(mn, nullptr);
    std::string modelName(pJstr);

    jchar *pJchar = env->GetCharArrayElements(charArr, nullptr);
    jint size = env->GetArrayLength(charArr);
    auto *data = (unsigned char *) pJchar;

    Openvino_Net *pNet = netPool.find(modelName)->second;

    jfloatArray outputDataArr = nullptr;
    if (pNet) {
        auto output = new Output();
        pNet->inference(*output, data, (int) w, (int) h);

        outputDataArr = env->NewFloatArray(output->getTotalDim());
        env->SetFloatArrayRegion(outputDataArr, 0, output->getTotalDim(), output->data);
        delete (output);
    }
    env->ReleaseCharArrayElements(charArr, pJchar, size);
    env->ReleaseStringUTFChars(mn, pJstr);
    return outputDataArr;
}

/*
 * Class:     com_mogujie_algo_openvino_jni_MoguOpenvino
 * Method:    release
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_mogujie_algo_openvino_jni_MoguOpenvino_release
        (JNIEnv *env, jclass, jstring jstr){

    std::string modelName;
    get_string(env, jstr, modelName);
    Openvino_Net *pNet = netPool.find(modelName)->second;
    if (pNet) {
        delete (pNet);
    }
}

