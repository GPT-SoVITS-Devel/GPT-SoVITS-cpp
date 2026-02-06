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
};

ONNXBackend::ONNXBackend() : impl_(std::make_unique<Impl>()) {}
ONNXBackend::~ONNXBackend() = default;

bool ONNXBackend::Load(const std::string& model_path, const Device& device,
                       int work_thread_num) {
  this->device_ = device;
  try {
    Ort::SessionOptions options;
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
    PrintInfo("[ONNXBackend] Loaded model from: {}", model_path);

    return true;
  } catch (const std::exception& e) {
    PrintError("[ONNXBackend] Load failed: {}", e.what());
    return false;
  }
}

namespace {
ONNXTensorElementDataType ToOnnxType(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::kFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::kInt32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::kInt64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::kInt8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::kUInt8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

DataType FromOnnxType(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return DataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return DataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return DataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return DataType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return DataType::kUInt8;
    default: THROW_ERROR("Unsupported ONNX data type");
  }
}
}

void ONNXBackend::Forward(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, Tensor*>& outputs) {
  
  Ort::MemoryInfo memory_info_cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  
  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_ort_values;

  for (auto const& [name, tensor] : inputs) {
    input_names.push_back(name.c_str());
    
    if (tensor->IsCPU()) {
        input_ort_values.push_back(Ort::Value::CreateTensor(
            memory_info_cpu, tensor->Data(), tensor->ByteSize(),
            tensor->Shape().data(), tensor->Shape().size(), ToOnnxType(tensor->Type())));
    } else {
#ifdef WITH_CUDA
        Ort::MemoryInfo memory_info_cuda("Cuda", OrtAllocatorType::OrtDeviceAllocator, device_.device_id, OrtMemTypeDefault);
        input_ort_values.push_back(Ort::Value::CreateTensor(
            memory_info_cuda, tensor->Data(), tensor->ByteSize(),
            tensor->Shape().data(), tensor->Shape().size(), ToOnnxType(tensor->Type())));
#else
        THROW_ERROR("CUDA tensor provided but CUDA backend not enabled");
#endif
    }
  }

  std::vector<std::string> requested_output_names;
  std::vector<Ort::AllocatedStringPtr> allocated_names; 
  if (outputs.empty()) {
      auto count = impl_->session->GetOutputCount();
      for (size_t i = 0; i < count; ++i) {
          auto name = impl_->session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
          requested_output_names.push_back(name.get());
          allocated_names.push_back(std::move(name));
      }
  } else {
      for (auto const& [name, _] : outputs) {
          requested_output_names.push_back(name);
      }
  }

  std::vector<const char*> output_names_ptrs;
  for (const auto& n : requested_output_names) output_names_ptrs.push_back(n.c_str());

  auto output_ort_values = impl_->session->Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_ort_values.data(),
      input_ort_values.size(), output_names_ptrs.data(), output_names_ptrs.size());

  for (size_t i = 0; i < output_names_ptrs.size(); ++i) {
      const char* name = output_names_ptrs[i];
      auto& val = output_ort_values[i];
      auto type_info = val.GetTensorTypeAndShapeInfo();
      auto shape = type_info.GetShape();
      auto dtype = FromOnnxType(type_info.GetElementType());

      auto mem_info = val.GetTensorMemoryInfo();
      Device actual_output_device(DeviceType::kCPU);
      
      if (mem_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU || 
          std::string(mem_info.GetAllocatorName()) == "Cuda") {
          actual_output_device = Device(DeviceType::kCUDA, mem_info.GetDeviceId());
      }

      auto* val_ptr = new Ort::Value(std::move(val));
      auto deleter = [val_ptr](void*) { delete val_ptr; };
      
      auto res_tensor = std::make_unique<Tensor>(
          val_ptr->GetTensorMutableData<void>(), 
          shape, dtype, actual_output_device, deleter);
      
      outputs[name] = res_tensor.release();
  }
}

std::vector<std::string> ONNXBackend::GetInputNames() const {
  return impl_->input_names;
}

std::vector<std::string> ONNXBackend::GetOutputNames() const {
  return impl_->output_names;
}

}  // namespace GPTSoVITS::Model