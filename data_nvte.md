
NVTETensor means TensorWrapper

TensorWrapper (c++ class)
    tensor_ (NVTETensor void*) // basically pointer to some sort of tensor index, think of uuid for tensor
    data (NVTEBasicTensor)
    amax (NVTEBasicTensor)
    scale (NVTEBasicTensor)
    scale_inv (NVTEBasicTensor)
        struct NVTEBasicTensor {
        void *data_ptr;
        NVTEDType dtype; // enum, e.g. kNVTEFloat8E8M0, kNVTEFloat4E2M1, kNVTEFloat32
        NVTEShape shape; // struct of ndim and data[15], both are size_t, unsigned long long
        };

enum NVTETensorParam {
  kNVTERowwiseData = 0,        /*!< Data usable in rowwise manner */
  kNVTEColumnwiseData = 1,     /*!< Data usable in columnwise manner */
  kNVTEScale = 2,              /*!< Scale tensor */
  kNVTEAmax = 3,               /*!< Amax tensor */
  kNVTERowwiseScaleInv = 4,    /*!< Scale inverse tensor for decoding Rowwise Data */
  kNVTEColumnwiseScaleInv = 5, /*!< Scale inverse tensor for decoding Columnwise Data */
  kNVTENumTensorParams
};

!!!TensorWrapper has method to set columnwise data, scale_inv but no memeber of them in the class
!!! internally, when these are set, it is actually setting on the struct below, which is an interpret cast of TensorWrapper to Tensor (struct) below

// reinterpret cast
Tensor (struct)
    SimpleTensor data;
    SimpleTensor columnwise_data;
    SimpleTensor amax;
    SimpleTensor scale;
    SimpleTensor scale_inv;
    SimpleTensor columnwise_scale_inv;
        struct SimpleTensor {
            void *dptr;
            std::vector<size_t> shape;
            DType dtype; // enum, e.g. kFloat8E4M3, kFloat4E2M1, 
    NVTEScalingMode scaling_mode;
    NVTETensor nvte_tensor; // pointer to nvte tensor registry

notice that there is convertNVTETensor
```c
void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param) {
  NVTE_CHECK(tensor != nullptr, "Tensor pointer can't be NULL.");
  auto *t = transformer_engine::convertNVTETensor(*tensor);
  NVTE_CHECK(t != nullptr, "Tensor is not allocated.");
  switch (param_name) {
    case kNVTERowwiseData:
      t->data = *param;
      break;
    case kNVTEColumnwiseData:
      t->columnwise_data = *param;
      break;
    case kNVTEScale:
      t->scale = *param;
      break;
    case kNVTEAmax:
      t->amax = *param;
      break;
    case kNVTERowwiseScaleInv:
      t->scale_inv = *param;
      break;
    case kNVTEColumnwiseScaleInv:
      t->columnwise_scale_inv = *param;
      break;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}
// ----------------------------------------------------------------
Tensor *convertNVTETensor(const NVTETensor t) {
  return TensorAllocator::instance().convertNVTETensor(t);
}

  Tensor *convertNVTETensor(NVTETensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    // 1-based indexing to enable 0-initialization of NVTETensor
    // to be invalid tensor
    static_assert(nullptr == 0);
    if (index != 0 && index <= size) {
      return &(memory[index - 1]);
    }
    return nullptr;
  }
```

!!! A tensor contains data and columnwise_data, 
many methods default to data attribute,
this means columnwise_data can only be used if data is not set,
otherwise data is used. what is the intent of this logic.
This would require the user explicitly set data to null if they want to use columnwise_data.
```c
  size_t dim() const {
    if (!has_data() && has_columnwise_data()) {
      return columnwise_data.shape.size();
    } else {
      return data.shape.size();
    }
  }
```
!!! Shape is following the same design.

Now, when pass the tensor from pytorch to cpp TE for gemm purpose,
pytorch tensor is wrapped with TensorWrapper,
NVTETensorFromMXFP8Tensor/NVTETensorFromNVFP4Tensor
What these function do: it sets the parameters of TensorWrapper, which is internally
setting the Tensor struct. The logic here is that any of rowwise_data or columnwise_data can be set, and both can be set at the same time. 
This will means shape logic will break, unless the user explicitly set data to null if they want to use columnwise_data. This can only be done in python side.
```c
TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(NVTE_MXFP8_1D_SCALING);

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  // Row-scaled data
  const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>();
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}
```


NVTETensorFromMXFP8Tensor called during make_transformer_engine_tensor, but quantizer is None,
quantizer->set_quantization_params(&ret); will enter NoneQuantizer's
void set_quantization_params(TensorWrapper* tensor) const override {} which does noting! omg why we design this way!?
when set_quantization_params is used? create_tensor

NVTETensorFromMXFP8Tensor is also only called before gemm


```c
    const DType fp8_dtype = tensor.attr("_fp8_dtype").cast<DType>(); // TODO(VS) factorize to _fp8_dtype, fp8_dtype
  if (rowwise_usage) {
    const auto &data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const auto &scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3,
                                 getTensorShape(scale_inv));
  }
```