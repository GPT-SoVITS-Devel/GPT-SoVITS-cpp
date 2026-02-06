//
// Created by 19254 on 2026/1/10.
//

#ifndef GPT_SOVITS_CPP_MODEL_BASE_H
#define GPT_SOVITS_CPP_MODEL_BASE_H

#include <string>
#include <vector>
#include <unordered_map>
#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/model/tensor.h"


namespace GPTSoVITS::Model {

/**
 * @brief 推理模型基类
 */
class BaseModel {
public:
    virtual ~BaseModel() = default;

    /**
     * @brief 加载模型文件
     * @param model_path 模型文件路径
     * @param device 运行设备
     * @param work_thread_num 工作线程数
     */
    virtual bool Load(const std::string& model_path, const Device& device,int work_thread_num) = 0;

    /**
     * @brief 执行推理
     * @param inputs 输入张量映射 (Name -> Tensor*)
     * @param outputs 输出张量映射 (Name -> Tensor*)
     */
    virtual void Forward(
        const std::unordered_map<std::string, Tensor*>& inputs,
        std::unordered_map<std::string, Tensor*>& outputs
    ) = 0;

    /**
     * @brief 获取所有输入节点的名称
     */
    virtual std::vector<std::string> GetInputNames() const = 0;

    /**
     * @brief 获取所有输出节点的名称
     */
    virtual std::vector<std::string> GetOutputNames() const = 0;

protected:
    Device device_;
};

} // namespace GPTSoVITS::Model


#endif //GPT_SOVITS_CPP_MODEL_BASE_H
