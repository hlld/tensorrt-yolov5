#include <iostream>
#include <chrono>
#include <map>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.6
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
// we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine_s(std::string& WeightFile, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WeightFile);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(128);
	//weightMap["deconv15"] = deconvwts15;

    ITensor* inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_m(std::string& WeightFile, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WeightFile);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 48, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 192, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 768, 768, 5, 9, 13, "model.8");
    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 384, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 384 * 2 * 2));
    for (int i = 0; i < 384 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 384 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 384, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(384);
    weightMap["deconv11"] = deconvwts11;
    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 768, 384, 2, false, 1, 0.5, "model.13");

    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 192, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 192 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 192, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(192);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 384, 192, 2, false, 1, 0.5, "model.17");

    //yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 192, 3, 2, 1, "model.18");

    ITensor* inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);

    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.20");

    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 384, 3, 2, 1, "model.21");

    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);

    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.23");

    // yolo layer 2
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_l(std::string& WeightFile, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WeightFile);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 64, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 128, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 256, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 256, 256, 9, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 512, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 512, 512, 9, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1024, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1024, 1024, 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 512, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 512 * 2 * 2));
    for (int i = 0; i < 512 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 512 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 512, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(512);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1024, 512, 3, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(256);
    ITensor* inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 512, 256, 3, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 256, 3, 2, 1, "model.18");

    ITensor* inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);

    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 512, 512, 3, false, 1, 0.5, "model.20");

    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 512, 3, 2, 1, "model.21");

    ITensor* inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);

    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.23");

    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine_x(std::string& WeightFile, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WeightFile);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 80, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 160, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 160, 160, 4, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 320, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 320, 320, 12, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 640, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 640, 640, 12, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1280, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1280, 1280, 5, 9, 13, "model.8");

    /* ------- yolov5 head ------- */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 640, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 640 * 2 * 2));
    for (int i = 0; i < 640 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 640 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 640, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(640);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1280, 640, 4, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 320, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 320 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 320, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(320);
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 640, 320, 4, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 320, 3, 2, 1, "model.18");

    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);

    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 640, 640, 4, false, 1, 0.5, "model.20");

    // yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 640, 3, 2, 1, "model.21");

    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);

    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.23");

    // yolo layer 2
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = { det2->getOutput(0), det1->getOutput(0), det0->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

std::string ParseModelPath(std::string& modelType, bool parseEngine) {
    std::string modelPath;

    if (modelType == "s") {
        modelPath = "../yolov5sv3";
    }
    else if (modelType == "m") {
        modelPath = "../yolov5mv3";
    }
    else if (modelType == "l") {
        modelPath = "../yolov5lv3";
    }
    else if (modelType == "x") {
        modelPath = "../yolov5xv3";
    }
    else {
        std::cerr << "unsupported model type " << modelType << std::endl;
        return modelPath;
    }

    if (parseEngine) {
        modelPath += ".engine";
    }
    else  {
        modelPath += ".wts";
    }
    return modelPath;
}

void APIToModel(std::string& modelType, unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = nullptr;
    std::string weightFile = ParseModelPath(modelType, false);
	if (modelType == "s") {
		engine = createEngine_s(weightFile, maxBatchSize, builder, config, DataType::kFLOAT);
	}
	else if (modelType == "m") {
		engine = createEngine_m(weightFile, maxBatchSize, builder, config, DataType::kFLOAT);
	}
	else if (modelType == "l") {
		engine = createEngine_l(weightFile, maxBatchSize, builder, config, DataType::kFLOAT);
	}
	else if (modelType == "x") {
		engine = createEngine_x(weightFile, maxBatchSize, builder, config, DataType::kFLOAT);
	}
	else {
		std::cerr << "unsupported model type " << modelType << std::endl;
	}
    // Create model to populate the network, then set the outputs and create an engine
    //ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::string classNameById(int classId) {
    static std::map<int, std::string> coco_ids;

    if (coco_ids.empty()) {
        coco_ids[0] = "person"; coco_ids[1] = "bicycle"; coco_ids[2] = "car"; coco_ids[3] = "motorcycle"; coco_ids[4] = "airplane";
        coco_ids[5] = "bus"; coco_ids[6] = "train"; coco_ids[7] = "truck"; coco_ids[8] = "boat"; coco_ids[9] = "traffic light";
        coco_ids[10] = "fire hydrant"; coco_ids[11] = "stop sign"; coco_ids[12] = "parking meter"; coco_ids[13] = "bench"; coco_ids[14] = "bird";
        coco_ids[15] = "cat"; coco_ids[16] = "dog"; coco_ids[17] = "horse"; coco_ids[18] = "sheep"; coco_ids[19] = "cow"; 
        coco_ids[20] = "elephant"; coco_ids[21] = "bear"; coco_ids[22] = "zebra"; coco_ids[23] = "giraffe"; coco_ids[24] = "backpack";
        coco_ids[25] = "umbrella"; coco_ids[26] = "handbag"; coco_ids[27] = "tie"; coco_ids[28] = "suitcase"; coco_ids[29] = "frisbee"; 
        coco_ids[30] = "skis"; coco_ids[31] = "snowboard"; coco_ids[32] = "sports ball"; coco_ids[33] = "kite"; coco_ids[34] = "baseball bat";
        coco_ids[35] = "baseball glove"; coco_ids[36] = "skateboard"; coco_ids[37] = "surfboard"; coco_ids[38] = "tennis racket"; coco_ids[39] = "bottle"; 
        coco_ids[40] = "wine glass"; coco_ids[41] = "cup"; coco_ids[42] = "fork"; coco_ids[43] = "knife"; coco_ids[44] = "spoon"; 
        coco_ids[45] = "bowl"; coco_ids[46] = "banana"; coco_ids[47] = "apple"; coco_ids[48] = "sandwich"; coco_ids[49] = "orange";
        coco_ids[50] = "broccoli"; coco_ids[51] = "carrot"; coco_ids[52] = "hot dog"; coco_ids[53] = "pizza"; coco_ids[54] = "donut"; 
        coco_ids[55] = "cake"; coco_ids[56] = "chair"; coco_ids[57] = "couch"; coco_ids[58] = "potted plant"; coco_ids[59] = "bed"; 
        coco_ids[60] = "dining table"; coco_ids[61] = "toilet"; coco_ids[62] = "tv"; coco_ids[63] = "laptop"; coco_ids[64] = "mouse"; 
        coco_ids[65] = "remote"; coco_ids[66] = "keyboard"; coco_ids[67] = "cell phone"; coco_ids[68] = "microwave"; coco_ids[69] = "oven";
        coco_ids[70] = "toaster"; coco_ids[71] = "sink"; coco_ids[72] = "refrigerator"; coco_ids[73] = "book"; coco_ids[74] = "clock"; 
        coco_ids[75] = "vase"; coco_ids[76] = "scissors"; coco_ids[77] = "teddy bear"; coco_ids[78] = "hair drier"; coco_ids[79] = "toothbrush";
    }

    if (classId < 0 || classId >= 80) {
        return std::string("background");
    }
    else {
        return coco_ids[classId];
    }
}

cv::Scalar classColorById(int classId) {
    const int num_classes = 80;
    static int color_tables[num_classes][3];
    static bool is_valid = false;

    if (!is_valid) {
        srand(0);
        for (int k = 0; k < num_classes; k++) {
            color_tables[k][0] = rand() % 256;
            color_tables[k][1] = rand() % 256;
            color_tables[k][2] = rand() % 256;
        }
        is_valid = true;
    }

    if (classId < 0 || classId >= num_classes) {
        return cv::Scalar(255, 255, 255);
    }
    else {
        return cv::Scalar(color_tables[classId][0], color_tables[classId][1], color_tables[classId][2]);
    }
}

void drawOutputs(cv::Mat& image, std::vector<Yolo::Detection>& results) {
    for (size_t k = 0; k < results.size(); k++) {
        cv::Rect rect = get_rect(image, results[k].bbox);
        cv::Scalar color = classColorById(int(results[k].class_id));
        int thick = int(0.7 * (image.rows + image.cols) / 700.F);
        cv::rectangle(image, rect, color, thick);

        std::string id_str = classNameById(int(results[k].class_id));
        char buffer[20] = {0};
        sprintf(buffer, "%s %.2f", id_str.data(), results[k].conf);
        std::string text(buffer);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int baseline;
        cv::Size size = cv::getTextSize(text, fontFace, fontScale, thick / 2, &baseline);
        cv::Rect rect_t(rect.x, rect.y - (size.height + 8), size.width + 4, size.height + 8);
        cv::rectangle(image, rect_t, color, -1);
        cv::putText(image, text, cv::Point(rect.x, rect.y - 5), fontFace, fontScale, 
			cv::Scalar(255, 255, 255), thick / 2);
    }
}

void preprocessCopy(cv::Mat& img, float* data_ptr) {
    cv::Mat pr_img = preprocess_img(img);
    int k = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data_ptr[k] = (float)uc_pixel[2] / 255.0;
            data_ptr[k + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data_ptr[k + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++k;
        }
    }
}

void yoloDetect(IExecutionContext* context, cv::VideoCapture& capture) {
    assert(BATCH_SIZE == 1);
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float outputs[BATCH_SIZE * OUTPUT_SIZE];

    cv::Mat frame;
    while(true) {
        auto start_time = std::chrono::system_clock::now();
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        preprocessCopy(frame, data);
        doInference(*context, data, outputs, BATCH_SIZE);
        std::vector<Yolo::Detection> results;
        nms(results, &outputs[0 * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        drawOutputs(frame, results);
        auto stop_time = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
        float duration_sec = float(duration.count()) * std::chrono::microseconds::period::num / 
            std::chrono::microseconds::period::den;
        float FPS = 1.F / duration_sec;

        char buffer[20] = {0};
        sprintf(buffer, "FPS: %d", int(FPS));
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        cv::putText(frame, std::string(buffer), cv::Point(20, 30), fontFace, fontScale, 
            cv::Scalar(255, 255, 255), 2);

        cv::imshow("video", frame);
        if (char(cv::waitKey(1)) == 'q') {
            break;
        }
    }
}

void cameraDetect(IExecutionContext* context, int cam_id) {
    cv::VideoCapture capture(cam_id);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
    std::cout << "opening camera " << cam_id << " ..." << std::endl;
    yoloDetect(context, capture);
}

void videoDetect(IExecutionContext* context, std::string& video_file) {
    cv::VideoCapture capture(video_file);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    capture.set(cv::CAP_PROP_FPS, 30);
    std::cout << "opening video " << video_file << " ..." << std::endl;
    yoloDetect(context, capture);
}

char* readEngineFile(std::string& engineFile, size_t& size) {
    char* stream_ptr = nullptr;
    std::ifstream file(engineFile, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        stream_ptr = new char[size];
        file.read(stream_ptr, size);
        file.close();
    }
    assert(stream_ptr);
    return stream_ptr;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    int detectMode = 1; // detection using images
    if (argc == 3 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
		std::string modelType(argv[2]);
        APIToModel(modelType, BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::string engineFile = ParseModelPath(modelType, true);
        std::ofstream p(engineFile, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 5 && std::string(argv[1]) == "-e" && std::string(argv[3]) == "-d") {
        std::string modelType(argv[2]);
        std::string engineFile = ParseModelPath(modelType, true);
        trtModelStream = readEngineFile(engineFile, size);
    } else if (argc == 5 && std::string(argv[1]) == "-e" && std::string(argv[3]) == "-v") {
        std::string modelType(argv[2]);
        std::string engineFile = ParseModelPath(modelType, true);
        trtModelStream = readEngineFile(engineFile, size);
        detectMode = 2; // detection using video file
    } else if (argc == 5 && std::string(argv[1]) == "-e" && std::string(argv[3]) == "-c") {
        std::string modelType(argv[2]);
        std::string engineFile = ParseModelPath(modelType, true);
        trtModelStream = readEngineFile(engineFile, size);
        detectMode = 3; // detection using camera stream
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s s // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -e s -d ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    if (detectMode == 2) {
        std::string video_file(argv[4]);
        videoDetect(context, video_file);
    }
    else if (detectMode == 3) {
        int cam_id = atoi(argv[4]);
        cameraDetect(context, cam_id);
    } else {
        std::vector<std::string> file_names;
        if (read_files_in_dir(argv[4], file_names) < 0) {
            std::cout << "read_files_in_dir failed." << std::endl;
            return -1;
        }

        // prepare input data ---------------------------
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];

        int fcount = 0;
        for (int f = 0; f < (int)file_names.size(); f++) {
            fcount++;
            if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
            for (int b = 0; b < fcount; b++) {
                cv::Mat img = cv::imread(std::string(argv[4]) + "/" + file_names[f - fcount + 1 + b]);
                if (img.empty()) continue;
                cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }

            // Run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, data, prob, BATCH_SIZE);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            }
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                //std::cout << res.size() << std::endl;
                cv::Mat img = cv::imread(std::string(argv[4]) + "/" + file_names[f - fcount + 1 + b]);
                for (size_t j = 0; j < res.size(); j++) {
                    cv::Rect r = get_rect(img, res[j].bbox);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), 
                        cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
                cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
            }
            fcount = 0;
        }
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
