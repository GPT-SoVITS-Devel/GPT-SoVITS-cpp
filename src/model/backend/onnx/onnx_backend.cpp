#include "GPTSoVITS/model/backend/onnx_backend.h"

#include <onnxruntime_cxx_api.h>

#include <iostream>

#include "GPTSoVITS/Text/Coding.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

struct ONNXBackend::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "gsv_cpp_bert"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, DataType> input_types;
  std::unordered_map<std::string, DataType> output_types;
};

ONNXBackend::ONNXBackend() : impl_(std::make_unique<Impl>()) {}
ONNXBackend::~ONNXBackend() = default;

namespace {
ONNXTensorElementDataType ToOnnxType(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::kInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

DataType FromOnnxType(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::kUInt8;
    default:
      THROW_ERROR("Unsupported ONNX data type");
  }
}

}  // namespace

std::vector<std::unique_ptr<Tensor>> ONNXBackend::InferCore(
  ONNXBackend::Impl* impl,
  const Device& device,
  const std::unordered_map<std::string, Tensor*>& inputs,
  const std::vector<std::string>& target_output_names) {

  Ort::IoBinding io_binding(*impl->session);

  // bind
  for (auto const& [name, tensor] : inputs) {
    Ort::MemoryInfo input_mem_info(nullptr);

    if (tensor->IsCPU()) {
      input_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    } else {
#ifdef WITH_CUDA
      input_mem_info = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                                       tensor->GetDevice().device_id, OrtMemTypeDefault);
#else
      THROW_ERROR("CUDA input provided but compiled without CUDA support.");
#endif
    }

    Ort::Value input_ort_value = Ort::Value::CreateTensor(
        input_mem_info, tensor->Data(), tensor->ByteSize(),
        tensor->Shape().data(), tensor->Shape().size(),
        ToOnnxType(tensor->Type()));

    io_binding.BindInput(name.c_str(), input_ort_value);
  }

  Ort::MemoryInfo output_mem_info(nullptr);
  if (device.type == DeviceType::kCPU) {
    output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  } else {
#ifdef WITH_CUDA
    output_mem_info = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                                      device.device_id, OrtMemTypeDefault);
#else
    output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
  }

  // output names
  for (const auto& name : target_output_names) {
    io_binding.BindOutput(name.c_str(), output_mem_info);
  }

  impl->session->Run(Ort::RunOptions{nullptr}, io_binding);

  auto result_values = io_binding.GetOutputValues();
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  output_tensors.reserve(result_values.size());

  for (size_t i = 0; i < result_values.size(); ++i) {
    auto& val = result_values[i];

    auto type_info = val.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    auto dtype = FromOnnxType(type_info.GetElementType());

    auto mem_info = val.GetTensorMemoryInfo();
    Device res_device(DeviceType::kCPU);

    if (mem_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU ||
        std::string(mem_info.GetAllocatorName()) == "Cuda") {
      res_device = Device(DeviceType::kCUDA, mem_info.GetDeviceId());
    }

    // 转移 Ort::Value 所有权给 Tensor
    // Ort::Value 不支持拷贝，必须移动到堆上由 Deleter 管理
    auto* value_holder = new Ort::Value(std::move(val));
    auto deleter = [value_holder](void*) { delete value_holder; };
    void* data_ptr = value_holder->GetTensorMutableData<void>();

    output_tensors.push_back(
        std::make_unique<Tensor>(data_ptr, shape, dtype, res_device, deleter));
  }

  return output_tensors;
}

bool ONNXBackend::Load(const std::string& model_path, const Device& device,
                       int work_thread_num) {
  this->device_ = device;
  try {
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(0);
    options.SetIntraOpNumThreads(work_thread_num);
    options.SetInterOpNumThreads(work_thread_num);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _ENABLE_CUDA_
    // 开启CUDA
    if (device.type == DeviceType::kCUDA) {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = device.device_id;
      options.AppendExecutionProvider_CUDA(cuda_options);
    }
#endif
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, Text::Utf8ToWstring(model_path).c_str(), options);

    // 收集输入信息
    auto input_count = impl_->session->GetInputCount();
    for (size_t i = 0; i < input_count; ++i) {
      auto name = impl_->session->GetInputNameAllocated(
          i, Ort::AllocatorWithDefaultOptions());
      std::string name_str = name.get();
      impl_->input_names.push_back(name_str);

      auto type_info = impl_->session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      impl_->input_types[name_str] = FromOnnxType(tensor_info.GetElementType());
    }

    // 收集输出信息
    auto output_count = impl_->session->GetOutputCount();
    for (size_t i = 0; i < output_count; ++i) {
      auto name = impl_->session->GetOutputNameAllocated(
          i, Ort::AllocatorWithDefaultOptions());
      std::string name_str = name.get();
      impl_->output_names.push_back(name_str);

      auto type_info = impl_->session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      impl_->output_types[name_str] =
          FromOnnxType(tensor_info.GetElementType());
    }

    PrintInfo("[ONNXBackend] Loaded model from: {}", model_path);

    return true;
  } catch (const std::exception& e) {
    PrintError("[ONNXBackend] Load failed: {}", e.what());
    return false;
  }
}

void ONNXBackend::Forward(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) {
  std::vector<std::string> output_names_req;

  if (outputs.empty()) {
    output_names_req = impl_->output_names;
  } else {
    for (const auto& [name, _] : outputs) {
      output_names_req.push_back(name);
    }
  }

  auto result_list = InferCore(impl_.get(), device_, inputs, output_names_req);

  // 结果回填
  for (size_t i = 0; i < result_list.size(); ++i) {
    outputs[output_names_req[i]] = std::move(result_list[i]);
  }
}

void ONNXBackend::Forward(const std::unordered_map<std::string, Tensor*>& inputs,
             std::vector<std::unique_ptr<Tensor>>& outputs) {
  outputs.clear();
  auto result_list = InferCore(impl_.get(), device_, inputs, impl_->output_names);
  outputs = std::move(result_list);
}

std::vector<std::string> ONNXBackend::GetInputNames() const {
  return impl_->input_names;
}

std::vector<std::string> ONNXBackend::GetOutputNames() const {
  return impl_->output_names;
}

DataType ONNXBackend::GetInputDataType(const std::string& name) const {
  auto it = impl_->input_types.find(name);
  if (it == impl_->input_types.end()) THROW_ERRORN("Input not found: {}", name);
  return it->second;
}

DataType ONNXBackend::GetOutputDataType(const std::string& name) const {
  auto it = impl_->output_types.find(name);
  if (it == impl_->output_types.end())
    THROW_ERRORN("Output not found: {}", name);
  return it->second;
}

}  // namespace GPTSoVITS::Model