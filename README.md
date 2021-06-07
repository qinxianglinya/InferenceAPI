# InferenceAPI

## 1、框架简介

本框架基于TensorRT进行开发，用于部署基于深度学习的目标检测算法，加速模型推理。该框架目前支持三种模型的部署：基于tensorflow的RetinaNet算法、基于Detectron2的RetinaNet算法、基于MMdetection的Yolov3算法。其中前两个算法已经正式落地，投入生产，第三个算法目前只进行了初步的测试，尚未正式投入生产。**说明：**由于第一次是使用基于Tensorflow的RetinaNet算法进行检测，因此本框架集成的第一个算法便是基于Tensorflow的RetinaNet，后由于Tensorflow可读性不高，对算法做优化比较困难，便采用基于Detectron2的RetinaNet算法做目标检测，本框架集成的第二个算法则是基于Detectron2的RetinaNet。后面考虑Yolov3检测速度和检测性能都要优于RetinaNet，便对Yolov3算法进行了部署，这也是本框架集成的第三个算法。

本框架提供**两个统一的API供外部调用**，API内部根据模型名称，自动调用对应的算法进行检测。用户如果需要集成自定义算法，根据框架内的集成规则，便可非常简单地将自定义算法集成进去，且对外的调用接口保持不变。

**该框架与modelTranslation软件配合使用**，modelTranslation软件的地址：。

modelTranslation软件用于将上述提到的三种目标检测模型转换为TensorRT engine，并反序列化至本地。本框架根据TensorRT engine，进行深度学习推理。

**为什么要将推理框架与模型转换软件（modelTranslation）分开实现？**

答：1）模型转换软件主要功能是将训练好的模型转换为TensorRT engine，本质是对模型进行优化。该部分的功能由深度学习相关从业人员进行维护。

​		2）推理框架更聚焦于从流程上来优化推理速度，该部分可以由普通程序开发人员进行优化。比如：开发者可以开多线程来调用多GPU，以此来加快推理速度；可以使用流水线技术，将CPU操作和GPU操作进行并行，来优化GPU等待CPU传输数据的时间；等等。

**因此，将两个功能进行独立**，开发出一个TensorRT模型转换软件和模型推理框架。对于模型转换软件使用者而言，在对深度学习没有丝毫了解的情况下，可以在根据可视化界面，根据要求填写参数，即可将训练好的模型转换为TensorRT engine；对于API调用者而言，其无需知道实现细节，只需根据约定好的接口，传入正确的数据，便可得到正确的结果。便于不同人员进行使用。

该框架目前支持：

- **多GPU推理**
- **FP32/FP16精度推理**
- **多batch推理**
- **Tensorflow-RetinaNet/Detectron2-RetinaNet/Detectron2-Yolov3**

## 2、使用方法

使用本框架提供的接口位于：trtInference.h头文件中。（**由于调用该API的检测程序是基于VS2005开发的，因此以下两个API的参数类型、及结构体的名称均按照约定的方式进行设置，用户可根据需求进行改写**）

**第一个API：**

```c++
static bool initEngines(const char* modelName);
```

**该接口用于实例化TensorRT Engine对象，TensorRT Engine对象用于模型推理。该对象对使用者透明的。用户在改变推理模型时，需要重新调用该函数，来重新实例化Engine对象。该函数内部在实例化Engine对象前，先判断Engine对象是否存在，如果存在，则释放原来Engine对象的内存，并重新实例化Engine。使用者在调用时无需关注旧Engine对象的内存释放问题。**

**该API自动检测显卡的数量，根据显卡的数量实例化多个Engine对象，用于多GPU调用。**

**参数说明：**const char* modelName: modelName是TensorRT模型的本地路径。

**返回类型：**bool: 如果实例化Engine对象成功，则返回True，否则，返回False。

**第二个API：**

```c++
static bool doInference(imgData::_SHARE_IMAGE_DATA* input, const int &totalNum, imgData::iDEFECT **result, bool isSynchronized = true);
```

**该接口用于使用Engine进行推理运算。如果存在多GPU，自动使用多GPU进行并行推断。**

**参数说明：**

- imgData::_SHARE_IMAGE_DATA* input：input表示输入的一组图片的指针，用于后续推理。
- const int &totalNum：表示图片的数量。
- imgData::iDEFECT **result：存放检测结果。第一维表示每张图片的索引，第二维表示每张图片的检测框。
- bool isSynchronized = true：基于CudaStream的同步推断，异步推断尚未实现。
- 返回类型：bool：如果检测成功，返回True，否则返回False。

## 3、环境配置

**硬件环境：**GeForce RTX 3070 * 2

**软件环境：**

- IDE: Visual Studio 2017
- TensorRT: 7.2.1.6
- Cuda11.1 + cudnn7.6.5

## 4、测试方案

该项目配合modelTranslation项目进行测试。

功能测试一：...

后续补充。

## 5、测试结果

后续补充。

## 6、回顾：部署过程中碰到的问题及解决思路

后续补充。

