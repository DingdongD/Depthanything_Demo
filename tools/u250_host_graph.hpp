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

  py::tuple gelu_pack_bf16_concatenate(
      const py::list &physical_inputs, const py::list &raw_source_descriptors,
      const py::dict &raw_target_descriptor, float scale) {
    if (physical_inputs.empty() ||
        physical_inputs.size() != raw_source_descriptors.size())
      throw std::invalid_argument(
          "physical GELU fusion requires aligned non-empty inputs and descriptors");
    require_scale(scale);

    std::vector<LayoutDescriptor> sources;
    sources.reserve(raw_source_descriptors.size());
    for (const py::handle &handle : raw_source_descriptors) {
      if (!py::isinstance<py::dict>(handle))
        throw std::invalid_argument("source descriptor must be a dictionary");
      sources.push_back(parse_descriptor(py::reinterpret_borrow<py::dict>(handle)));
    }
    const LayoutDescriptor target = parse_descriptor(raw_target_descriptor);
    if (target.layout != "NDWC" || target.bitdepth != 8 ||
        target.direction != "input")
      throw std::invalid_argument(
          "physical GELU fusion target must be an INT8 NDWC input");

    size_t concatenated_channels = 0;
    std::vector<std::array<py::array, 2>> source_arrays;
    std::vector<std::array<const uint8_t *, 2>> source_banks;
    source_arrays.reserve(sources.size());
    source_banks.reserve(sources.size());
    for (size_t index = 0; index < sources.size(); ++index) {
      const LayoutDescriptor &source = sources[index];
      if (source.layout != "NDWC" || source.bitdepth != 16 ||
          source.direction != "output")
        throw std::invalid_argument(
            "physical GELU fusion sources must be BF16 NDWC outputs");
      for (size_t axis = 0; axis < 3; ++axis)
        if (source.dims[axis] != target.dims[axis])
          throw std::invalid_argument(
              "physical GELU fusion source extents do not match target");
      concatenated_channels += source.dims[3];

      if (!py::isinstance<py::tuple>(physical_inputs[index]))
        throw std::invalid_argument("physical input must be an (even, odd) pair");
      const py::tuple pair = py::reinterpret_borrow<py::tuple>(physical_inputs[index]);
      if (pair.size() != 2)
        throw std::invalid_argument("physical input must be an (even, odd) pair");
      std::array<py::array, 2> arrays = {
          py::array::ensure(pair[0]), py::array::ensure(pair[1])};
      const size_t expected = source.combined_bytes / 2;
      for (size_t bank = 0; bank < 2; ++bank) {
        if (!arrays[bank])
          throw std::invalid_argument("physical input bank must be a NumPy array");
        require_array(arrays[bank], py::dtype::of<uint8_t>(), "physical input bank");
        if (arrays[bank].ndim() != 1 ||
            static_cast<size_t>(arrays[bank].size()) != expected)
          throw std::invalid_argument(
              "physical input bank size does not match source descriptor");
      }
      source_banks.push_back({
          static_cast<const uint8_t *>(arrays[0].data()),
          static_cast<const uint8_t *>(arrays[1].data())});
      source_arrays.push_back(std::move(arrays));
    }
    if (concatenated_channels != target.dims[3])
      throw std::invalid_argument(
          "physical GELU fusion channels do not sum to target channels");

    const size_t half = target.combined_bytes / 2;
    py::array_t<uint8_t> even(half), odd(half);
    uint8_t *target_banks[] = {even.mutable_data(), odd.mutable_data()};
    const size_t rows = target.dims[0] * target.dims[1] * target.dims[2];
    const auto lut = gelu_lut(scale, target.elements);
    std::atomic<bool> nonfinite{false};
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      std::memset(target_banks[0], 0, half);
      std::memset(target_banks[1], 0, half);
      parallel_rows(rows, target.dims[3], [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
          const size_t n = row / (target.dims[1] * target.dims[2]);
          const size_t d = (row / target.dims[2]) % target.dims[1];
          const size_t w = row % target.dims[2];
          size_t channel_offset = 0;
          for (size_t source_index = 0; source_index < sources.size(); ++source_index) {
            const LayoutDescriptor &source = sources[source_index];
            for (size_t c = 0; c < source.dims[3]; ++c) {
              const BankOffset source_location =
                  matrix_physical_index(source, n, d, w, c);
              const uint8_t *source_bank =
                  source_banks[source_index][source_location.bank];
              const uint16_t bits =
                  static_cast<uint16_t>(source_bank[source_location.offset]) |
                  (static_cast<uint16_t>(source_bank[source_location.offset + 1]) << 8U);
              if ((bits & 0x7f80U) == 0x7f80U)
                nonfinite.store(true, std::memory_order_relaxed);
              const BankOffset target_location =
                  matrix_physical_index(target, n, d, w, channel_offset + c);
              target_banks[target_location.bank][target_location.offset] = (*lut)[bits];
            }
            channel_offset += source.dims[3];
          }
        }
      });
    }
    if (nonfinite.load(std::memory_order_relaxed))
      throw std::invalid_argument("host executor requires finite BF16 values");
    record(Kind::GeluPackBf16Concatenate, begin, target.elements);
    return py::make_tuple(std::move(even), std::move(odd));
  }

  py::tuple attention_pack_bf16_heads(
      const py::list &physical_inputs, const py::list &raw_source_descriptors,
      const py::list &raw_valid_widths, const py::dict &raw_target_descriptor,
      float scale, size_t heads) {
    if (physical_inputs.empty() || heads == 0 ||
        physical_inputs.size() != raw_source_descriptors.size() ||
        physical_inputs.size() != raw_valid_widths.size() ||
        physical_inputs.size() % heads != 0)
      throw std::invalid_argument("physical attention fusion inputs are not aligned");
    require_scale(scale);
    const LayoutDescriptor target = parse_descriptor(raw_target_descriptor);
    if (target.layout != "NDWC" || target.bitdepth != 8 ||
        target.direction != "input" || target.dims[3] % heads != 0)
      throw std::invalid_argument(
          "physical attention fusion target must be a head-aligned INT8 NDWC input");
    const size_t head_width = target.dims[3] / heads;
    const size_t chunks_per_head = physical_inputs.size() / heads;
    std::vector<LayoutDescriptor> sources;
    std::vector<size_t> valid_widths;
    std::vector<std::array<py::array, 2>> source_arrays;
    std::vector<std::array<const uint8_t *, 2>> source_banks;
    sources.reserve(physical_inputs.size());
    valid_widths.reserve(physical_inputs.size());
    source_arrays.reserve(physical_inputs.size());
    source_banks.reserve(physical_inputs.size());
    for (size_t index = 0; index < physical_inputs.size(); ++index) {
      if (!py::isinstance<py::dict>(raw_source_descriptors[index]))
        throw std::invalid_argument("source descriptor must be a dictionary");
      LayoutDescriptor source = parse_descriptor(
          py::reinterpret_borrow<py::dict>(raw_source_descriptors[index]));
      if (source.layout != "NDWC" || source.bitdepth != 16 ||
          source.direction != "output" || source.dims[0] != target.dims[0] ||
          source.dims[1] != target.dims[1] || source.dims[3] != head_width)
        throw std::invalid_argument(
            "physical attention source does not match target head geometry");
      const long long raw_width = py::cast<long long>(raw_valid_widths[index]);
      if (raw_width <= 0 || static_cast<size_t>(raw_width) > source.dims[2])
        throw std::invalid_argument("physical attention valid width is out of range");
      const py::tuple pair = py::cast<py::tuple>(physical_inputs[index]);
      if (pair.size() != 2)
        throw std::invalid_argument("physical input must be an (even, odd) pair");
      std::array<py::array, 2> arrays = {
          py::array::ensure(pair[0]), py::array::ensure(pair[1])};
      const size_t expected = source.combined_bytes / 2;
      for (size_t bank = 0; bank < 2; ++bank) {
        if (!arrays[bank])
          throw std::invalid_argument("physical input bank must be a NumPy array");
        require_array(arrays[bank], py::dtype::of<uint8_t>(), "physical input bank");
        if (arrays[bank].ndim() != 1 ||
            static_cast<size_t>(arrays[bank].size()) != expected)
          throw std::invalid_argument(
              "physical input bank size does not match source descriptor");
      }
      sources.push_back(std::move(source));
      valid_widths.push_back(static_cast<size_t>(raw_width));
      source_banks.push_back({
          static_cast<const uint8_t *>(arrays[0].data()),
          static_cast<const uint8_t *>(arrays[1].data())});
      source_arrays.push_back(std::move(arrays));
    }

    std::vector<size_t> source_for(heads * target.dims[2]);
    std::vector<size_t> source_width(heads * target.dims[2]);
    for (size_t head = 0; head < heads; ++head) {
      size_t target_width = 0;
      for (size_t chunk = 0; chunk < chunks_per_head; ++chunk) {
        const size_t source_index = head * chunks_per_head + chunk;
        for (size_t w = 0; w < valid_widths[source_index]; ++w) {
          if (target_width >= target.dims[2])
            throw std::invalid_argument("physical attention chunks exceed target width");
          source_for[head * target.dims[2] + target_width] = source_index;
          source_width[head * target.dims[2] + target_width] = w;
          ++target_width;
        }
      }
      if (target_width != target.dims[2])
        throw std::invalid_argument("physical attention chunks do not cover target width");
    }

    const size_t half = target.combined_bytes / 2;
    py::array_t<uint8_t> even(half), odd(half);
    uint8_t *target_banks[] = {even.mutable_data(), odd.mutable_data()};
    const size_t rows = target.dims[0] * target.dims[1] * target.dims[2];
    const auto lut = bf16_quantize_lut(scale, target.elements);
    std::atomic<bool> nonfinite{false};
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      std::memset(target_banks[0], 0, half);
      std::memset(target_banks[1], 0, half);
      parallel_rows(rows, target.dims[3], [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
          const size_t n = row / (target.dims[1] * target.dims[2]);
          const size_t d = (row / target.dims[2]) % target.dims[1];
          const size_t w = row % target.dims[2];
          for (size_t head = 0; head < heads; ++head) {
            const size_t map = head * target.dims[2] + w;
            const size_t source_index = source_for[map];
            const size_t sw = source_width[map];
            for (size_t c = 0; c < head_width; ++c) {
              const BankOffset source_location = matrix_physical_index(
                  sources[source_index], n, d, sw, c);
              const uint8_t *source_bank =
                  source_banks[source_index][source_location.bank];
              const uint16_t bits =
                  static_cast<uint16_t>(source_bank[source_location.offset]) |
                  (static_cast<uint16_t>(source_bank[source_location.offset + 1]) << 8U);
              if ((bits & 0x7f80U) == 0x7f80U)
                nonfinite.store(true, std::memory_order_relaxed);
              const BankOffset target_location = matrix_physical_index(
                  target, n, d, w, head * head_width + c);
              target_banks[target_location.bank][target_location.offset] = (*lut)[bits];
            }
          }
        }
      });
    }
    if (nonfinite.load(std::memory_order_relaxed))
      throw std::invalid_argument("host executor requires finite BF16 values");
    record(Kind::AttentionPackBf16Heads, begin, target.elements);
    return py::make_tuple(std::move(even), std::move(odd));
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

  py::array resize_align_corners(const py::array &input,
                                 py::ssize_t output_height,
                                 py::ssize_t output_width) {
    const py::buffer_info info = require_float32(input, "Resize input");
    if (info.ndim != 4)
      throw std::invalid_argument("Resize input must be rank-4 NCHW");
    if (output_height <= 0 || output_width <= 0)
      throw std::invalid_argument("Resize output dimensions must be positive");
    validate_finite(static_cast<const float *>(info.ptr),
                    static_cast<size_t>(info.size));

    const size_t batches = static_cast<size_t>(info.shape[0]);
    const size_t channels = static_cast<size_t>(info.shape[1]);
    const size_t input_height = static_cast<size_t>(info.shape[2]);
    const size_t input_width = static_cast<size_t>(info.shape[3]);
    const size_t out_height = static_cast<size_t>(output_height);
    const size_t out_width = static_cast<size_t>(output_width);
    std::vector<py::ssize_t> shape = {
        info.shape[0], info.shape[1], output_height, output_width};
    py::array_t<float> output(shape);
    const float *source = static_cast<const float *>(info.ptr);
    float *target = output.mutable_data();

    struct Coordinate {
      size_t lower;
      size_t upper;
      double weight;
    };
    const auto coordinates = [](size_t input_size, size_t output_size) {
      std::vector<Coordinate> result(output_size);
      for (size_t index = 0; index < output_size; ++index) {
        const float position = output_size > 1
            ? static_cast<float>(static_cast<double>(index) *
                static_cast<double>(input_size - 1) /
                static_cast<double>(output_size - 1))
            : 0.0f;
        const size_t lower = static_cast<size_t>(std::floor(position));
        result[index] = {
            lower, std::min(lower + 1, input_size - 1),
            static_cast<double>(position) - static_cast<double>(lower)};
      }
      return result;
    };
    const std::vector<Coordinate> ys = coordinates(input_height, out_height);
    const std::vector<Coordinate> xs = coordinates(input_width, out_width);
    const size_t rows = checked_multiply(
        checked_multiply(batches, channels, "Resize batch/channel extent"),
        out_height, "Resize output row extent");
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      parallel_rows(rows, out_width, [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
          const size_t output_y = row % out_height;
          const size_t plane = row / out_height;
          const float *plane_source = source + plane * input_height * input_width;
          float *row_target = target + row * out_width;
          const Coordinate &y = ys[output_y];
          for (size_t output_x = 0; output_x < out_width; ++output_x) {
            const Coordinate &x = xs[output_x];
            const double upper_left = plane_source[y.lower * input_width + x.lower];
            const double lower_left = plane_source[y.upper * input_width + x.lower];
            const double upper_right = plane_source[y.lower * input_width + x.upper];
            const double lower_right = plane_source[y.upper * input_width + x.upper];
            const double vertical_left =
                upper_left * (1.0 - y.weight) + lower_left * y.weight;
            const double vertical_right =
                upper_right * (1.0 - y.weight) + lower_right * y.weight;
            row_target[output_x] = static_cast<float>(
                vertical_left * (1.0 - x.weight) + vertical_right * x.weight);
          }
        }
      });
    }
    record(Kind::ResizeAlignCorners, begin,
           checked_multiply(rows, out_width, "Resize output extent"));
    return output;
  }

  py::dict stats() const {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    py::dict result;
    result["host_calls"] = total_calls();
    result["quantize_calls"] = quantize_calls_;
    result["gelu_quantize_calls"] = gelu_quantize_calls_;
    result["gelu_pack_bf16_concatenate_calls"] =
        gelu_pack_bf16_concatenate_calls_;
    result["attention_pack_bf16_heads_calls"] = attention_pack_bf16_heads_calls_;
    result["gelu_lut_hits"] = gelu_lut_hits_;
    result["gelu_lut_misses"] = gelu_lut_misses_;
    result["gelu_lut_elements"] = gelu_lut_elements_;
    result["quantize_lut_hits"] = quantize_lut_hits_;
    result["quantize_lut_misses"] = quantize_lut_misses_;
    result["quantize_lut_elements"] = quantize_lut_elements_;
    result["add_calls"] = add_calls_;
    result["add_quantize_calls"] = add_quantize_calls_;
    result["concatenate_calls"] = concatenate_calls_;
    result["resize_align_corners_calls"] = resize_align_corners_calls_;
    result["host_elements"] = elements_;
    result["host_seconds"] = seconds_;
    return result;
  }

  void reset_stats() {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    quantize_calls_ = 0;
    gelu_quantize_calls_ = 0;
    gelu_pack_bf16_concatenate_calls_ = 0;
    attention_pack_bf16_heads_calls_ = 0;
    gelu_lut_hits_ = 0;
    gelu_lut_misses_ = 0;
    gelu_lut_elements_ = 0;
    quantize_lut_hits_ = 0;
    quantize_lut_misses_ = 0;
    quantize_lut_elements_ = 0;
    add_calls_ = 0;
    add_quantize_calls_ = 0;
    concatenate_calls_ = 0;
    resize_align_corners_calls_ = 0;
    elements_ = 0;
    seconds_ = 0.0;
  }

 private:
  enum class Kind {
    Quantize, GeluQuantize, GeluPackBf16Concatenate, AttentionPackBf16Heads,
    Add, AddQuantize,
    Concatenate, ResizeAlignCorners
  };

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

  std::shared_ptr<const std::array<int8_t, 65536>> bf16_quantize_lut(
      float scale, size_t elements) {
    uint32_t key;
    std::memcpy(&key, &scale, sizeof(key));
    std::shared_ptr<const std::array<int8_t, 65536>> result;
    bool hit = false;
    {
      std::lock_guard<std::mutex> guard(quantize_lut_cache_mutex());
      auto &cache = quantize_lut_cache();
      const auto found = cache.find(key);
      if (found != cache.end()) {
        result = found->second;
        hit = true;
      } else {
        auto created = std::make_shared<std::array<int8_t, 65536>>();
        for (size_t index = 0; index < created->size(); ++index) {
          const float value = fp32_from_bf16_bits(static_cast<uint16_t>(index));
          (*created)[index] = std::isfinite(value)
              ? quantize_scalar(value, scale) : 0;
        }
        result = created;
        cache.emplace(key, std::move(created));
      }
    }
    {
      std::lock_guard<std::mutex> guard(stats_mutex_);
      if (hit) ++quantize_lut_hits_;
      else ++quantize_lut_misses_;
      quantize_lut_elements_ += elements;
    }
    return result;
  }

  static std::mutex &quantize_lut_cache_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<uint32_t,
      std::shared_ptr<const std::array<int8_t, 65536>>> &quantize_lut_cache() {
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
      case Kind::GeluPackBf16Concatenate:
        ++gelu_pack_bf16_concatenate_calls_; break;
      case Kind::AttentionPackBf16Heads:
        ++attention_pack_bf16_heads_calls_; break;
      case Kind::Add: ++add_calls_; break;
      case Kind::AddQuantize: ++add_quantize_calls_; break;
      case Kind::Concatenate: ++concatenate_calls_; break;
      case Kind::ResizeAlignCorners: ++resize_align_corners_calls_; break;
    }
    elements_ += elements;
    seconds_ += elapsed;
  }

  uint64_t total_calls() const {
    return quantize_calls_ + gelu_quantize_calls_ +
        gelu_pack_bf16_concatenate_calls_ + add_calls_ +
        attention_pack_bf16_heads_calls_ +
        add_quantize_calls_ + concatenate_calls_ + resize_align_corners_calls_;
  }

  mutable std::mutex stats_mutex_;
  uint64_t quantize_calls_ = 0;
  uint64_t gelu_quantize_calls_ = 0;
  uint64_t gelu_pack_bf16_concatenate_calls_ = 0;
  uint64_t attention_pack_bf16_heads_calls_ = 0;
  uint64_t gelu_lut_hits_ = 0;
  uint64_t gelu_lut_misses_ = 0;
  uint64_t gelu_lut_elements_ = 0;
  uint64_t quantize_lut_hits_ = 0;
  uint64_t quantize_lut_misses_ = 0;
  uint64_t quantize_lut_elements_ = 0;
  uint64_t add_calls_ = 0;
  uint64_t add_quantize_calls_ = 0;
  uint64_t concatenate_calls_ = 0;
  uint64_t resize_align_corners_calls_ = 0;
  uint64_t elements_ = 0;
  double seconds_ = 0.0;
};
