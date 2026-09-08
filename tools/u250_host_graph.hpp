#pragma once

class HostGraphExecutor {
 public:
  HostGraphExecutor() = default;

  py::array quantize(const py::array &input, float scale) {
    const py::buffer_info info = require_float32(input, "quantize input");
    require_scale(scale);
    validate_finite(static_cast<const float *>(info.ptr),
                    static_cast<size_t>(info.size));
    py::array_t<int8_t> output(info.shape);
    const float *source = static_cast<const float *>(info.ptr);
    int8_t *target = output.mutable_data();
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      quantize_values(source, target, static_cast<size_t>(info.size), scale);
    }
    record(Kind::Quantize, begin, static_cast<size_t>(info.size));
    return output;
  }

  py::array gelu_quantize(const py::array &input, float scale) {
    const py::buffer_info info = require_float32(input, "GELU input");
    require_scale(scale);
    const float *source = static_cast<const float *>(info.ptr);
    const size_t count = static_cast<size_t>(info.size);
    const bool bf16_exact = validate_finite_and_bf16(source, count);
    py::array_t<int8_t> output(info.shape);
    int8_t *target = output.mutable_data();
    const auto begin = std::chrono::steady_clock::now();
    const auto lut = bf16_exact ? gelu_lut(scale, count) : nullptr;
    {
      py::gil_scoped_release release;
      parallel_rows(count, 1, [&](size_t row_begin, size_t row_end) {
        if (lut) {
          for (size_t index = row_begin; index < row_end; ++index)
            target[index] = (*lut)[bf16_bits(source[index])];
        } else {
          for (size_t index = row_begin; index < row_end; ++index)
            target[index] = quantize_scalar(gelu_scalar(source[index]), scale);
        }
      });
    }
    record(Kind::GeluQuantize, begin, count);
    return output;
  }

  py::array add(const py::array &left, const py::array &right) {
    const py::buffer_info left_info = require_float32(left, "add left");
    const py::buffer_info right_info = require_float32(right, "add right");
    require_same_shape(left_info, right_info);
    validate_finite(static_cast<const float *>(left_info.ptr),
                    static_cast<size_t>(left_info.size));
    validate_finite(static_cast<const float *>(right_info.ptr),
                    static_cast<size_t>(right_info.size));
    py::array_t<float> output(left_info.shape);
    const float *left_data = static_cast<const float *>(left_info.ptr);
    const float *right_data = static_cast<const float *>(right_info.ptr);
    float *target = output.mutable_data();
    const size_t count = static_cast<size_t>(left_info.size);
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      parallel_rows(count, 1, [&](size_t row_begin, size_t row_end) {
        for (size_t index = row_begin; index < row_end; ++index)
          target[index] = left_data[index] + right_data[index];
      });
    }
    record(Kind::Add, begin, count);
    return output;
  }

  py::array add_quantize(const py::array &left, const py::array &right,
                         float scale) {
    const py::buffer_info left_info = require_float32(left, "add left");
    const py::buffer_info right_info = require_float32(right, "add right");
    require_same_shape(left_info, right_info);
    require_scale(scale);
    validate_finite(static_cast<const float *>(left_info.ptr),
                    static_cast<size_t>(left_info.size));
    validate_finite(static_cast<const float *>(right_info.ptr),
                    static_cast<size_t>(right_info.size));
    py::array_t<int8_t> output(left_info.shape);
    const float *left_data = static_cast<const float *>(left_info.ptr);
    const float *right_data = static_cast<const float *>(right_info.ptr);
    int8_t *target = output.mutable_data();
    const size_t count = static_cast<size_t>(left_info.size);
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      parallel_rows(count, 1, [&](size_t row_begin, size_t row_end) {
        for (size_t index = row_begin; index < row_end; ++index) {
          const float value = left_data[index] + right_data[index];
          target[index] = quantize_scalar(value, scale);
        }
      });
    }
    record(Kind::AddQuantize, begin, count);
    return output;
  }

  py::array concatenate(const py::list &inputs, int axis) {
    if (inputs.empty())
      throw std::invalid_argument("concatenate inputs must be non-empty");
    std::vector<py::array> arrays;
    arrays.reserve(inputs.size());
    for (const py::handle &handle : inputs) {
      if (!py::isinstance<py::array>(handle))
        throw std::invalid_argument("concatenate inputs must be NumPy arrays");
      arrays.push_back(py::reinterpret_borrow<py::array>(handle));
    }
    const py::buffer_info first = require_supported_array(
        arrays.front(), "concatenate input");
    const int rank = first.ndim;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
      throw std::invalid_argument("concatenate axis is out of range");

    std::vector<py::buffer_info> infos;
    infos.reserve(arrays.size());
    std::vector<py::ssize_t> shape(first.shape.begin(), first.shape.end());
    shape[axis] = 0;
    for (const py::array &array : arrays) {
      py::buffer_info info = require_supported_array(array, "concatenate input");
      if (!array.dtype().is(arrays.front().dtype()) || info.ndim != rank)
        throw std::invalid_argument("concatenate dtype or rank mismatch");
      for (int dimension = 0; dimension < rank; ++dimension) {
        if (dimension != axis && info.shape[dimension] != first.shape[dimension])
          throw std::invalid_argument("concatenate shape mismatch");
      }
      shape[axis] += info.shape[axis];
      infos.push_back(std::move(info));
    }

    py::array output(arrays.front().dtype(), shape);
    const size_t element_bytes = static_cast<size_t>(first.itemsize);
    size_t outer = 1, inner = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
      outer = checked_multiply(outer, static_cast<size_t>(shape[dimension]),
                               "concatenate outer extent");
    for (int dimension = axis + 1; dimension < rank; ++dimension)
      inner = checked_multiply(inner, static_cast<size_t>(shape[dimension]),
                               "concatenate inner extent");
    uint8_t *target = static_cast<uint8_t *>(output.mutable_data());
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      parallel_rows(outer, inner * static_cast<size_t>(shape[axis]),
                    [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
          size_t output_axis_offset = 0;
          for (const py::buffer_info &info : infos) {
            const size_t axis_size = static_cast<size_t>(info.shape[axis]);
            const size_t bytes = axis_size * inner * element_bytes;
            const auto *source = static_cast<const uint8_t *>(info.ptr)
                + row * bytes;
            const size_t output_row_bytes =
                static_cast<size_t>(shape[axis]) * inner * element_bytes;
            std::memcpy(target + row * output_row_bytes + output_axis_offset,
                        source, bytes);
            output_axis_offset += bytes;
          }
        }
      });
    }
    record(Kind::Concatenate, begin, static_cast<size_t>(output.size()));
    return output;
  }

  py::dict stats() const {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    py::dict result;
    result["host_calls"] = total_calls();
    result["quantize_calls"] = quantize_calls_;
    result["gelu_quantize_calls"] = gelu_quantize_calls_;
    result["gelu_lut_hits"] = gelu_lut_hits_;
    result["gelu_lut_misses"] = gelu_lut_misses_;
    result["gelu_lut_elements"] = gelu_lut_elements_;
    result["add_calls"] = add_calls_;
    result["add_quantize_calls"] = add_quantize_calls_;
    result["concatenate_calls"] = concatenate_calls_;
    result["host_elements"] = elements_;
    result["host_seconds"] = seconds_;
    return result;
  }

  void reset_stats() {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    quantize_calls_ = 0;
    gelu_quantize_calls_ = 0;
    gelu_lut_hits_ = 0;
    gelu_lut_misses_ = 0;
    gelu_lut_elements_ = 0;
    add_calls_ = 0;
    add_quantize_calls_ = 0;
    concatenate_calls_ = 0;
    elements_ = 0;
    seconds_ = 0.0;
  }

 private:
  enum class Kind { Quantize, GeluQuantize, Add, AddQuantize, Concatenate };

  static py::buffer_info require_supported_array(const py::array &array,
                                                  const char *label) {
    if (!array.dtype().is(py::dtype::of<float>()) &&
        !array.dtype().is(py::dtype::of<int8_t>()))
      throw std::invalid_argument(std::string(label) +
                                  " must have float32 or int8 dtype");
    if (!(array.flags() & py::array::c_style))
      throw std::invalid_argument(std::string(label) + " must be C-contiguous");
    py::buffer_info info = array.request();
    if (info.ndim <= 0 || info.size <= 0)
      throw std::invalid_argument(std::string(label) + " must be non-empty");
    return info;
  }

  static py::buffer_info require_float32(const py::array &array,
                                         const char *label) {
    if (!array.dtype().is(py::dtype::of<float>()))
      throw std::invalid_argument(std::string(label) + " must be float32");
    return require_supported_array(array, label);
  }

  static void require_same_shape(const py::buffer_info &left,
                                 const py::buffer_info &right) {
    if (left.shape != right.shape)
      throw std::invalid_argument("add inputs must have identical shape");
  }

  static void require_scale(float scale) {
    if (!std::isfinite(scale) || !(scale > 0.0f))
      throw std::invalid_argument("quantization scale must be finite and positive");
  }

  static void validate_finite(const float *values, size_t count) {
    for (size_t index = 0; index < count; ++index)
      if (!std::isfinite(values[index]))
        throw std::invalid_argument("host executor requires finite float32 values");
  }

  static uint16_t bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16U);
  }

  static bool validate_finite_and_bf16(const float *values, size_t count) {
    bool exact = true;
    for (size_t index = 0; index < count; ++index) {
      if (!std::isfinite(values[index]))
        throw std::invalid_argument("host executor requires finite float32 values");
      uint32_t bits;
      std::memcpy(&bits, values + index, sizeof(bits));
      exact = exact && (bits & 0xffffU) == 0;
    }
    return exact;
  }

  static float fp32_from_bf16_bits(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  std::shared_ptr<const std::array<int8_t, 65536>> gelu_lut(
      float scale, size_t elements) {
    uint32_t key;
    std::memcpy(&key, &scale, sizeof(key));
    std::shared_ptr<const std::array<int8_t, 65536>> result;
    bool hit = false;
    {
      std::lock_guard<std::mutex> guard(gelu_lut_cache_mutex());
      auto &cache = gelu_lut_cache();
      const auto found = cache.find(key);
      if (found != cache.end()) {
        result = found->second;
        hit = true;
      } else {
        auto created = std::make_shared<std::array<int8_t, 65536>>();
        for (size_t index = 0; index < created->size(); ++index) {
          const float value = fp32_from_bf16_bits(static_cast<uint16_t>(index));
          (*created)[index] = std::isfinite(value)
              ? quantize_scalar(gelu_scalar(value), scale) : 0;
        }
        result = created;
        cache.emplace(key, std::move(created));
      }
    }
    {
      std::lock_guard<std::mutex> guard(stats_mutex_);
      if (hit) ++gelu_lut_hits_;
      else ++gelu_lut_misses_;
      gelu_lut_elements_ += elements;
    }
    return result;
  }

  static std::mutex &gelu_lut_cache_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<uint32_t,
      std::shared_ptr<const std::array<int8_t, 65536>>> &gelu_lut_cache() {
    static std::unordered_map<uint32_t,
        std::shared_ptr<const std::array<int8_t, 65536>>> cache;
    return cache;
  }

  static int8_t quantize_scalar(float value, float scale) {
    const float rounded = std::nearbyint(value / scale);
    const float clamped = std::max(-128.0f, std::min(127.0f, rounded));
    return static_cast<int8_t>(clamped);
  }

  static void quantize_values(const float *source, int8_t *target,
                              size_t count, float scale) {
    parallel_rows(count, 1, [&](size_t row_begin, size_t row_end) {
      for (size_t index = row_begin; index < row_end; ++index)
        target[index] = quantize_scalar(source[index], scale);
    });
  }

  static float gelu_scalar(float value) {
    constexpr float sqrt2 = 1.41421356237309504880f;
    const float x = value / sqrt2;
    const float sign = static_cast<float>((x > 0.0f) - (x < 0.0f));
    const float a = std::fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    const float erf = sign * (1.0f - polynomial * numpy_expf(-(a * a)));
    return 0.5f * value * (1.0f + erf);
  }

  static float numpy_expf(float value) {
    // Scalar transcription of NumPy 1.26's AVX2/FMA float-exp kernel.  The
    // deployed Python reference uses that ufunc, whose last bits can differ
    // from libm expf at INT8 half-way boundaries.
    // The normal-number range reduction below cannot construct subnormals:
    // decrementing the exponent field would wrap and turn a tiny result into
    // a large value.  In GELU this branch is already far below the point at
    // which exp() can affect the FP32 value of erf, so zero is bit-equivalent
    // to NumPy for the final INT8 result.
    if (value < -87.33654475f) return 0.0f;
    constexpr float log2e = 1.44269504088896340736f;
    constexpr float magic = 0x1.800000p+23f;
    float quadrant = value * log2e;
    quadrant = (quadrant + magic) - magic;
    float reduced = std::fma(quadrant, -6.93145752e-1f, value);
    reduced = std::fma(quadrant, -1.42860677e-6f, reduced);
    float numerator = std::fma(5.082762527590693718096e-04f, reduced,
                               6.757896990527504603057e-03f);
    numerator = std::fma(numerator, reduced, 5.114512081637298353406e-02f);
    numerator = std::fma(numerator, reduced, 2.473615434895520810817e-01f);
    numerator = std::fma(numerator, reduced, 7.257664613233124478488e-01f);
    numerator = std::fma(numerator, reduced, 9.999999999980870924916e-01f);
    float denominator = std::fma(2.159509375685829852307e-02f, reduced,
                                 -2.742335390411667452936e-01f);
    denominator = std::fma(denominator, reduced, 1.0f);
    float result = numerator / denominator;
    uint32_t bits;
    std::memcpy(&bits, &result, sizeof(bits));
    bits += static_cast<uint32_t>(static_cast<int32_t>(quadrant)) << 23;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  void record(Kind kind, std::chrono::steady_clock::time_point begin,
              size_t elements) {
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::lock_guard<std::mutex> guard(stats_mutex_);
    switch (kind) {
      case Kind::Quantize: ++quantize_calls_; break;
      case Kind::GeluQuantize: ++gelu_quantize_calls_; break;
      case Kind::Add: ++add_calls_; break;
      case Kind::AddQuantize: ++add_quantize_calls_; break;
      case Kind::Concatenate: ++concatenate_calls_; break;
    }
    elements_ += elements;
    seconds_ += elapsed;
  }

  uint64_t total_calls() const {
    return quantize_calls_ + gelu_quantize_calls_ + add_calls_ +
        add_quantize_calls_ + concatenate_calls_;
  }

  mutable std::mutex stats_mutex_;
  uint64_t quantize_calls_ = 0;
  uint64_t gelu_quantize_calls_ = 0;
  uint64_t gelu_lut_hits_ = 0;
  uint64_t gelu_lut_misses_ = 0;
  uint64_t gelu_lut_elements_ = 0;
  uint64_t add_calls_ = 0;
  uint64_t add_quantize_calls_ = 0;
  uint64_t concatenate_calls_ = 0;
  uint64_t elements_ = 0;
  double seconds_ = 0.0;
};
