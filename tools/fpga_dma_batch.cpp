#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

namespace {
constexpr const char *kH2C[2] = {"/dev/xdma0_h2c_0", "/dev/xdma0_h2c_1"};
constexpr const char *kC2H[2] = {"/dev/xdma0_c2h_0", "/dev/xdma0_c2h_1"};
constexpr const char *kUser = "/dev/xdma0_user";

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message + ": " + std::strerror(errno));
}

size_t align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

size_t ceil_div(size_t value, size_t divisor) {
  return (value + divisor - 1) / divisor;
}

size_t checked_multiply(size_t left, size_t right, const char *label) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    throw std::invalid_argument(std::string(label) + " overflows");
  return left * right;
}

struct LayoutDescriptor {
  std::string layout;
  std::array<size_t, 4> dims{};
  int bitdepth = 0;
  size_t c_align = 0;
  size_t w_align = 0;
  size_t combined_bytes = 0;
  std::string direction;
  size_t index = 0;
  std::string matrix_role;
  size_t elements = 0;
};

void validate_nchw_extent(const LayoutDescriptor &descriptor);

LayoutDescriptor parse_descriptor(const py::dict &descriptor) {
  const auto require = [&](const char *field) -> py::handle {
    if (!descriptor.contains(field))
      throw std::invalid_argument(std::string("descriptor is missing ") + field);
    return descriptor[field];
  };
  const auto nonnegative_size = [&](const char *field,
                                    const char *negative_message) -> size_t {
    const long long value = py::cast<long long>(require(field));
    if (value < 0) throw std::invalid_argument(negative_message);
    return static_cast<size_t>(value);
  };

  LayoutDescriptor result;
  try {
    result.layout = py::cast<std::string>(require("layout"));
    result.bitdepth = py::cast<int>(require("bitdepth"));
    result.c_align = nonnegative_size("c_align", "tensor alignment must be positive");
    result.w_align = nonnegative_size("w_align", "tensor alignment must be positive");
    result.combined_bytes = nonnegative_size(
        "combined_bytes", "combined extent must be 256-byte aligned");
    result.direction = py::cast<std::string>(require("direction"));
    result.index = nonnegative_size("index", "descriptor index must be non-negative");
    if (descriptor.contains("matrix_role"))
      result.matrix_role = py::cast<std::string>(descriptor["matrix_role"]);
  } catch (const py::cast_error &) {
    throw std::invalid_argument("invalid tensor descriptor field type");
  }

  const py::handle raw_dims = require("dims");
  if (py::isinstance<py::str>(raw_dims) || !py::isinstance<py::sequence>(raw_dims))
    throw std::invalid_argument("tensor dims must have rank four");
  const py::sequence dims = py::reinterpret_borrow<py::sequence>(raw_dims);
  if (dims.size() != 4)
    throw std::invalid_argument("tensor dims must have rank four");
  for (size_t axis = 0; axis < result.dims.size(); ++axis) {
    try {
      const long long value = py::cast<long long>(dims[axis]);
      if (value <= 0) throw std::invalid_argument("tensor dims must be positive");
      result.dims[axis] = static_cast<size_t>(value);
    } catch (const py::cast_error &) {
      throw std::invalid_argument("tensor dims must be integers");
    }
  }

  if (result.layout != "NCHW" && result.layout != "NDWC")
    throw std::invalid_argument("unsupported layout " + result.layout);
  if (result.bitdepth != 8 && result.bitdepth != 16)
    throw std::invalid_argument("unsupported bitdepth " + std::to_string(result.bitdepth));
  if (result.c_align == 0 || result.w_align == 0)
    throw std::invalid_argument("tensor alignment must be positive");
  if (result.direction != "input" && result.direction != "output")
    throw std::invalid_argument("invalid tensor direction " + result.direction);
  if (result.matrix_role.empty()) {
    if (result.layout == "NCHW") {
      result.matrix_role = "netio";
    } else if (result.direction == "output") {
      result.matrix_role = "output";
    } else {
      throw std::invalid_argument("NDWC input descriptor requires matrix_role");
    }
  }
  if (result.matrix_role != "netio" && result.matrix_role != "left" &&
      result.matrix_role != "right" && result.matrix_role != "output")
    throw std::invalid_argument("unsupported matrix_role " + result.matrix_role);
  const bool matching_matrix_role =
      (result.layout == "NCHW" && result.matrix_role == "netio") ||
      (result.layout == "NDWC" && result.direction == "input" &&
       (result.matrix_role == "left" || result.matrix_role == "right")) ||
      (result.layout == "NDWC" && result.direction == "output" &&
       result.matrix_role == "output");
  if (!matching_matrix_role)
    throw std::invalid_argument("matrix_role does not match layout and direction");
  if (result.combined_bytes == 0 || result.combined_bytes % 256 != 0)
    throw std::invalid_argument("combined extent must be 256-byte aligned");

  result.elements = 1;
  for (const size_t dim : result.dims)
    result.elements = checked_multiply(result.elements, dim, "tensor element count");
  const size_t logical_bytes = checked_multiply(
      result.elements, static_cast<size_t>(result.bitdepth / 8), "logical tensor size");
  if (result.combined_bytes < logical_bytes)
    throw std::invalid_argument("combined extent is smaller than logical tensor");
  if (result.layout == "NCHW") validate_nchw_extent(result);
  return result;
}

py::dict normalized_descriptor(const py::dict &descriptor) {
  const LayoutDescriptor parsed = parse_descriptor(descriptor);
  py::dict result;
  result["layout"] = parsed.layout;
  py::list dims;
  for (const size_t dim : parsed.dims) dims.append(dim);
  result["dims"] = dims;
  result["bitdepth"] = parsed.bitdepth;
  result["c_align"] = parsed.c_align;
  result["w_align"] = parsed.w_align;
  result["combined_bytes"] = parsed.combined_bytes;
  result["direction"] = parsed.direction;
  result["index"] = parsed.index;
  result["matrix_role"] = parsed.matrix_role;
  result["half_bytes"] = parsed.combined_bytes / 2;
  result["elements"] = parsed.elements;
  return result;
}

uint16_t fp32_to_bf16_rne(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("BF16 conversion requires finite float32 values");
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + rounding) >> 16U);
}

[[maybe_unused]] float bf16_to_fp32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t test_fp32_to_bf16(const py::handle &value) {
  const py::array scalar = py::array::ensure(value);
  if (!scalar || scalar.ndim() != 0 || !scalar.dtype().is(py::dtype::of<float>()))
    throw py::type_error("_test_fp32_to_bf16 requires a float32 scalar");
  float fp32;
  std::memcpy(&fp32, scalar.data(), sizeof(fp32));
  return fp32_to_bf16_rne(fp32);
}

template <typename Function>
void parallel_rows(size_t rows, size_t work_per_row, Function function) {
  const unsigned available = std::max(1U, std::thread::hardware_concurrency());
  size_t workers = std::min<size_t>({8, rows, static_cast<size_t>(available)});
  if (workers <= 1 || rows * work_per_row < 65536) {
    function(0, rows);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (size_t worker = 1; worker < workers; ++worker) {
    const size_t begin = rows * worker / workers;
    const size_t end = rows * (worker + 1) / workers;
    threads.emplace_back([=, &function] { function(begin, end); });
  }
  function(0, rows / workers);
  for (auto &thread : threads) thread.join();
}

struct NchwShape {
  size_t n = 0, c = 0, h = 0, w = 0;
};

NchwShape nchw_shape(const py::buffer_info &info) {
  if (info.ndim != 4) throw std::invalid_argument("INT8 codec requires NCHW rank 4");
  NchwShape result{static_cast<size_t>(info.shape[0]),
                   static_cast<size_t>(info.shape[1]),
                   static_cast<size_t>(info.shape[2]),
                   static_cast<size_t>(info.shape[3])};
  if (result.n == 0 || result.c == 0 || result.h == 0 || result.w == 0)
    throw std::invalid_argument("INT8 codec does not accept empty tensors");
  return result;
}

std::string layout_kind(const NchwShape &shape, size_t physical_bytes) {
  const size_t rows = checked_multiply(shape.n, shape.h, "NCHW physical extent");
  const size_t normal = checked_multiply(
      checked_multiply(rows, ceil_div(shape.w, 16), "NCHW physical extent"),
      checked_multiply(ceil_div(shape.c, 16), 256, "NCHW physical extent"),
      "NCHW physical extent");
  if (normal == physical_bytes) return "normal16";
  const size_t compact = checked_multiply(
      checked_multiply(rows, ceil_div(shape.w, 64), "NCHW physical extent"),
      256, "NCHW physical extent");
  if (shape.c <= 4 && compact == physical_bytes) return "compact4";
  return {};
}

size_t normal16_half_size(const NchwShape &shape) {
  return shape.n * shape.h * ceil_div(shape.w, 16) *
         ceil_div(shape.c, 16) * 128;
}

size_t normal16_index(const NchwShape &shape, size_t n, size_t c,
                      size_t y, size_t x) {
  const size_t width_blocks = ceil_div(shape.w, 16);
  const size_t channel_blocks = ceil_div(shape.c, 16);
  return (((((n * shape.h + y) * width_blocks + x / 16) *
             channel_blocks + c / 16) * 8 + x % 8) * 16 + c % 16);
}

size_t compact4_index(const NchwShape &shape, size_t n, size_t c,
                      size_t y, size_t x) {
  return (((((n * shape.h + y) * ceil_div(shape.w, 64) + x / 64) * 8
             + x % 8) * 4 + (x % 64) / 16) * 4 + c);
}

void validate_nchw_extent(const LayoutDescriptor &descriptor) {
  const auto &d = descriptor.dims;
  const NchwShape shape{d[0], d[1], d[2], d[3]};
  const size_t element_bytes = descriptor.bitdepth / 8;
  const std::string kind = layout_kind(shape, descriptor.combined_bytes / element_bytes);
  if (kind.empty())
    throw std::invalid_argument("NCHW physical extent does not match padded normal16/compact4 storage");
  // DS cfg alignments are physical strides in 16-byte words, not axis quanta.
  const size_t c_stride = checked_multiply(
      kind == "normal16" ? ceil_div(shape.c, 16) : 1,
      element_bytes, "NCHW channel alignment");
  const size_t w_stride = checked_multiply(
      ceil_div(shape.w, kind == "normal16" ? 16 : 64), c_stride,
      "NCHW width alignment");
  if (descriptor.c_align != c_stride || descriptor.w_align != w_stride)
    throw std::invalid_argument("NCHW alignment strides do not match physical extent");
  if (descriptor.combined_bytes > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max()))
    throw std::invalid_argument("NCHW physical extent exceeds NumPy size limit");
}

struct BankOffset { size_t bank, offset; };

BankOffset nchw_byte_offset(const NchwShape &shape, bool compact, size_t element_bytes,
                           size_t n, size_t c, size_t y, size_t x) {
  const size_t lane = compact ? compact4_index(shape, n, c, y, x)
                              : normal16_index(shape, n, c, y, x);
  // Expand the established INT8 lane into combined bytes before splitting the
  // 128-byte DDR stripes. BF16 therefore changes banks at width 4, not width 8.
  const size_t combined_lane = (lane / 128) * 256 + ((x % 16) / 8) * 128 + lane % 128;
  const size_t byte = combined_lane * element_bytes;
  return {(byte / 128) % 2, (byte / 256) * 128 + byte % 128};
}

void require_array(const py::array &array, const py::dtype &dtype, const char *label) {
  if (!array.dtype().is(dtype))
    throw std::invalid_argument(std::string(label) + " dtype is incompatible with descriptor");
  if (!(array.flags() & py::array::c_style))
    throw std::invalid_argument(std::string(label) + " must be C-contiguous");
}

py::tuple pack_tensor(const py::array &input, const py::dict &raw_descriptor) {
  const LayoutDescriptor descriptor = parse_descriptor(raw_descriptor);
  if (descriptor.layout != "NCHW")
    throw std::invalid_argument("pack_tensor does not yet support NDWC");
  require_array(input, descriptor.bitdepth == 8 ? py::dtype::of<int8_t>()
                                               : py::dtype::of<float>(), "input");
  if (input.ndim() != 4)
    throw std::invalid_argument("input shape does not match descriptor");
  for (size_t axis = 0; axis < 4; ++axis)
    if (static_cast<size_t>(input.shape(axis)) != descriptor.dims[axis])
      throw std::invalid_argument("input shape does not match descriptor");
  const auto &d = descriptor.dims;
  const NchwShape shape{d[0], d[1], d[2], d[3]};
  const size_t element_bytes = descriptor.bitdepth / 8;
  const bool compact = layout_kind(shape, descriptor.combined_bytes / element_bytes) == "compact4";
  const auto *source = static_cast<const uint8_t *>(input.data());
  // Validate before launching worker threads: exceptions cannot escape a worker.
  if (descriptor.bitdepth == 16) {
    for (size_t i = 0; i < descriptor.elements; ++i) {
      float value;
      std::memcpy(&value, source + i * sizeof(float), sizeof(value));
      if (!std::isfinite(value))
        throw std::invalid_argument("BF16 conversion requires finite float32 values");
    }
  }
  const size_t half = descriptor.combined_bytes / 2;
  py::array_t<uint8_t> even(half), odd(half);
  uint8_t *banks[] = {even.mutable_data(), odd.mutable_data()};
  {
    py::gil_scoped_release release;
    std::memset(banks[0], 0, half);
    std::memset(banks[1], 0, half);
    parallel_rows(shape.n * shape.h, shape.c * shape.w, [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        const size_t n = row / shape.h, y = row % shape.h;
        for (size_t c = 0; c < shape.c; ++c) {
          for (size_t x = 0; x < shape.w; ++x) {
            const size_t logical = ((n * shape.c + c) * shape.h + y) * shape.w + x;
            const BankOffset target = nchw_byte_offset(shape, compact, element_bytes, n, c, y, x);
            uint16_t bits = source[logical];
            if (element_bytes == 2) {
              float value;
              std::memcpy(&value, source + logical * sizeof(float), sizeof(value));
              bits = fp32_to_bf16_rne(value);
            }
            banks[target.bank][target.offset] = static_cast<uint8_t>(bits);
            if (element_bytes == 2)
              banks[target.bank][target.offset + 1] = static_cast<uint8_t>(bits >> 8);
          }
        }
      }
    });
  }
  return py::make_tuple(std::move(even), std::move(odd));
}

py::array unpack_tensor(const py::array &even, const py::array &odd,
                        const py::dict &raw_descriptor) {
  const LayoutDescriptor descriptor = parse_descriptor(raw_descriptor);
  if (descriptor.layout != "NCHW")
    throw std::invalid_argument("unpack_tensor does not yet support NDWC");
  require_array(even, py::dtype::of<uint8_t>(), "even bank");
  require_array(odd, py::dtype::of<uint8_t>(), "odd bank");
  const size_t half = descriptor.combined_bytes / 2;
  if (even.ndim() != 1 || odd.ndim() != 1 ||
      static_cast<size_t>(even.size()) != half || static_cast<size_t>(odd.size()) != half)
    throw std::invalid_argument("bank buffer sizes must equal descriptor combined_bytes / 2");
  const auto &d = descriptor.dims;
  const NchwShape shape{d[0], d[1], d[2], d[3]};
  const size_t element_bytes = descriptor.bitdepth / 8;
  const bool compact = layout_kind(shape, descriptor.combined_bytes / element_bytes) == "compact4";
  py::array output(descriptor.bitdepth == 8 ? py::dtype::of<int8_t>() : py::dtype::of<float>(),
                   {d[0], d[1], d[2], d[3]});
  auto *target = static_cast<uint8_t *>(output.mutable_data());
  const uint8_t *banks[] = {static_cast<const uint8_t *>(even.data()),
                            static_cast<const uint8_t *>(odd.data())};
  {
    py::gil_scoped_release release;
    parallel_rows(shape.n * shape.h, shape.c * shape.w, [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        const size_t n = row / shape.h, y = row % shape.h;
        for (size_t c = 0; c < shape.c; ++c) {
          for (size_t x = 0; x < shape.w; ++x) {
            const BankOffset source = nchw_byte_offset(shape, compact, element_bytes, n, c, y, x);
            const size_t logical = ((n * shape.c + c) * shape.h + y) * shape.w + x;
            if (element_bytes == 1) {
              target[logical] = banks[source.bank][source.offset];
            } else {
              const uint16_t bits = banks[source.bank][source.offset] |
                  (static_cast<uint16_t>(banks[source.bank][source.offset + 1]) << 8);
              const float value = bf16_to_fp32(bits);
              std::memcpy(target + logical * sizeof(float), &value, sizeof(value));
            }
          }
        }
      }
    });
  }
  return output;
}

using Normal16Pair = std::array<
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>, 2>;

Normal16Pair normal16_pair(const py::handle &handle, const NchwShape &shape,
                           const char *label) {
  const py::tuple pair = py::cast<py::tuple>(handle);
  if (pair.size() != 2)
    throw std::invalid_argument(std::string(label) + " must be an (even,odd) pair");
  const size_t expected = normal16_half_size(shape);
  Normal16Pair result;
  for (size_t bank = 0; bank < 2; ++bank) {
    result[bank] = py::array_t<uint8_t, py::array::c_style |
                               py::array::forcecast>::ensure(pair[bank]);
    if (!result[bank] || static_cast<size_t>(result[bank].size()) != expected)
      throw std::invalid_argument(std::string(label) +
                                  " does not match its normal16 shape");
  }
  return result;
}

class LockedBuffer {
 public:
  LockedBuffer() = default;
  ~LockedBuffer() { reset(); }
  LockedBuffer(const LockedBuffer &) = delete;
  LockedBuffer &operator=(const LockedBuffer &) = delete;

  void reserve(size_t requested) {
    if (requested <= capacity_) return;
    reset();
    capacity_ = align_up(std::max<size_t>(requested, 4096), 4096);
    if (::posix_memalign(reinterpret_cast<void **>(&data_), 4096, capacity_) != 0)
      fail("posix_memalign");
    std::memset(data_, 0, capacity_);
    if (::mlock(data_, capacity_) != 0) {
      const int saved_errno = errno;
      std::free(data_);
      data_ = nullptr;
      capacity_ = 0;
      errno = saved_errno;
      fail("mlock");
    }
  }

  uint8_t *data() { return data_; }
  size_t capacity() const { return capacity_; }

 private:
  void reset() {
    if (!data_) return;
    ::munlock(data_, capacity_);
    std::free(data_);
    data_ = nullptr;
    capacity_ = 0;
  }
  uint8_t *data_ = nullptr;
  size_t capacity_ = 0;
};

struct Segment {
  int bank = 0;
  uint64_t address = 0;
  size_t size = 0;
  size_t scratch_offset = 0;
  size_t result_index = 0;
};

void seek_write_exact(int fd, const uint8_t *data, size_t size, uint64_t address) {
  if (::lseek(fd, static_cast<off_t>(address), SEEK_SET) < 0) fail("XDMA H2C lseek");
  size_t done = 0;
  size_t interrupted = 0;
  while (done < size) {
    const ssize_t result = ::write(fd, data + done, size - done);
    if (result < 0 && errno == EINTR && interrupted++ < 16)
      continue;
    if (result < 0) fail("XDMA H2C write");
    if (result == 0) throw std::runtime_error("short XDMA H2C write");
    done += static_cast<size_t>(result);
  }
}

void seek_read_exact(int fd, uint8_t *data, size_t size, uint64_t address) {
  if (::lseek(fd, static_cast<off_t>(address), SEEK_SET) < 0) fail("XDMA C2H lseek");
  size_t done = 0;
  size_t interrupted = 0;
  while (done < size) {
    const ssize_t result = ::read(fd, data + done, size - done);
    if (result < 0 && errno == EINTR && interrupted++ < 16)
      continue;
    if (result < 0) fail("XDMA C2H read");
    if (result == 0) throw std::runtime_error("short XDMA C2H read");
    done += static_cast<size_t>(result);
  }
}

class DmaBatch {
 public:
  DmaBatch() {
    for (int bank = 0; bank < 2; ++bank) {
      h2c_[bank] = ::open(kH2C[bank], O_RDWR);
      if (h2c_[bank] < 0) fail(std::string("open ") + kH2C[bank]);
      c2h_[bank] = ::open(kC2H[bank], O_RDWR | O_NONBLOCK);
      if (c2h_[bank] < 0) fail(std::string("open ") + kC2H[bank]);
    }
    user_fd_ = ::open(kUser, O_RDWR);
    if (user_fd_ < 0) fail(std::string("open ") + kUser);
    user_mapping_ = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                           MAP_SHARED, user_fd_, 0);
    if (user_mapping_ == MAP_FAILED) fail("mmap XDMA user BAR");
    for (int index = 0; index < 16; ++index) {
      const std::string path = "/dev/xdma0_events_" + std::to_string(index);
      events_[index] = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
      if (events_[index] < 0) fail("open " + path);
    }
  }

  ~DmaBatch() {
    for (int fd : events_) if (fd >= 0) ::close(fd);
    if (user_mapping_ != MAP_FAILED) ::munmap(user_mapping_, 4096);
    if (user_fd_ >= 0) ::close(user_fd_);
    for (int fd : h2c_) if (fd >= 0) ::close(fd);
    for (int fd : c2h_) if (fd >= 0) ::close(fd);
  }

  void h2c_batch(const py::list &requests) { h2c_batch_impl(requests, false); }
  void h2c_batch_safe(const py::list &requests) { h2c_batch_impl(requests, true); }

  void h2c_batch_impl(const py::list &requests, bool reopen_each_segment) {
    std::array<std::vector<Segment>, 2> segments;
    std::array<size_t, 2> cursor{};
    std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>> arrays;
    for (const py::handle &handle : requests) {
      py::tuple item = py::cast<py::tuple>(handle);
      if (item.size() != 3) throw std::invalid_argument("H2C request must be (bank,address,array)");
      const int bank = py::cast<int>(item[0]);
      if (bank < 0 || bank > 1) throw std::invalid_argument("bank must be 0 or 1");
      const uint64_t address = py::cast<uint64_t>(item[1]);
      auto array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(item[2]);
      if (!array) throw std::invalid_argument("H2C payload must be a contiguous uint8 array");
      const size_t size = static_cast<size_t>(array.size());
      if (size == 0) throw std::invalid_argument("zero-byte H2C request");
      const size_t offset = cursor[bank];
      cursor[bank] += size;
      const size_t array_index = arrays.size();
      arrays.push_back(std::move(array));
      segments[bank].push_back(Segment{bank, address, size, offset, array_index});
    }
    for (int bank = 0; bank < 2; ++bank) input_[bank].reserve(cursor[bank]);
    for (int bank = 0; bank < 2; ++bank) {
      for (const Segment &segment : segments[bank]) {
        std::memcpy(input_[bank].data() + segment.scratch_offset,
                    arrays[segment.result_index].data(), segment.size);
      }
    }
    run_parallel([&](int bank) {
      write_segments(bank, segments[bank], reopen_each_segment);
    });
    if (reopen_each_segment) ++safe_h2c_batches_;
    ++h2c_batches_;
    h2c_segments_ += requests.size();
    h2c_bytes_ += cursor[0] + cursor[1];
  }

  py::list c2h_batch(const py::list &requests) {
    return c2h_batch_impl(requests, false);
  }
  py::list c2h_batch_safe(const py::list &requests) {
    return c2h_batch_impl(requests, true);
  }

  py::list run_npu_chain(const py::list &programs, int timeout_ms) {
    if (programs.empty()) throw std::invalid_argument("NPU chain is empty");
    if (timeout_ms <= 0) throw std::invalid_argument("timeout must be positive");
    std::lock_guard<std::mutex> guard(schedule_mutex_);
    py::list timings;
    for (const py::handle &handle : programs) {
      const py::dict program = py::cast<py::dict>(handle);
      const double elapsed = launch_program(program, timeout_ms);
      timings.append(elapsed);
    }
    ++npu_chain_calls_;
    return timings;
  }

  py::dict run_cbam_fused_pool(const py::list &source_pairs,
                               const py::dict &avg_program,
                               const py::dict &max_program,
                               const py::dict &fc_program,
                               const py::dict &layout,
                               int timeout_ms) {
    // Execute the whole dec2 CBAM global-pool/FC1 slice as one Python->C++
    // transaction.  The two C32 source tensors occupy disjoint DDR
    // workspaces, so both input bank pairs are submitted in one H2C batch and
    // reused by average and maximum reduction.  Only the two 32-byte logical
    // reduction vectors cross PCIe for the C32+C32 -> C64 exact-span join.
    if (source_pairs.size() != 2)
      throw std::invalid_argument("CBAM fused pool requires exactly two C32 inputs");
    if (timeout_ms <= 0) throw std::invalid_argument("timeout must be positive");

    const uint64_t workspace = py::cast<uint64_t>(layout["workspace_base_blocks"]);
    const uint64_t reduce_extent = py::cast<uint64_t>(layout["reduce_extent_blocks"]);
    const uint64_t reduce_input = py::cast<uint64_t>(layout["reduce_input_addr"]);
    const uint64_t reduce_output = py::cast<uint64_t>(layout["reduce_output_addr"]);
    const size_t reduce_input_size = py::cast<size_t>(layout["reduce_input_size"]);
    const size_t reduce_output_size = py::cast<size_t>(layout["reduce_output_size"]);
    const uint64_t fc_input = py::cast<uint64_t>(layout["fc_input_addr"]);
    const uint64_t fc_output = py::cast<uint64_t>(layout["fc_output_addr"]);
    const size_t fc_input_size = py::cast<size_t>(layout["fc_input_size"]);
    const size_t fc_output_size = py::cast<size_t>(layout["fc_output_size"]);
    if (reduce_extent == 0 || reduce_input_size == 0 || reduce_output_size == 0 ||
        fc_input_size != reduce_output_size * 2 || fc_output_size == 0)
      throw std::invalid_argument("invalid CBAM exact-span layout");
    if (reduce_output < reduce_input + ceil_div(reduce_input_size, 128))
      throw std::invalid_argument("CBAM reducer output overlaps reusable input");

    std::array<Normal16Pair, 2> sources{
        normal16_pair(source_pairs[0], NchwShape{1, 32, 228, 304}, "CBAM segment0"),
        normal16_pair(source_pairs[1], NchwShape{1, 32, 228, 304}, "CBAM segment1")};
    for (const auto &pair : sources)
      for (const auto &bank : pair)
        if (static_cast<size_t>(bank.size()) != reduce_input_size)
          throw std::invalid_argument("CBAM source physical size mismatch");

    // Fused multi-op ISA is not fully position-independent outside its
    // compiler-qualified FM split.  Reuse the average carrier's resident FM
    // region for both reductions (their physical I/O ABI is identical), and
    // the FC carrier's own resident FM region for both FC launches.
    const std::vector<uint64_t> avg_resident_bases =
        py::cast<std::vector<uint64_t>>(avg_program["base_addresses"]);
    const std::vector<uint64_t> max_resident_bases =
        py::cast<std::vector<uint64_t>>(max_program["base_addresses"]);
    const std::vector<uint64_t> fc_resident_bases =
        py::cast<std::vector<uint64_t>>(fc_program["base_addresses"]);
    if (avg_resident_bases.size() != 6 || max_resident_bases.size() != 6 ||
        fc_resident_bases.size() != 6)
      throw std::invalid_argument("CBAM resident base vector is malformed");
    const uint64_t avg_reduce_base = avg_resident_bases[4];
    const uint64_t max_reduce_base = max_resident_bases[4];
    const uint64_t avg_fc_base = fc_resident_bases[4];
    const uint64_t max_fc_base = fc_resident_bases[4];
    auto with_fm_base = [](const py::dict &source, uint64_t fm_base) {
      py::dict result(source);
      std::vector<uint64_t> bases =
          py::cast<std::vector<uint64_t>>(source["base_addresses"]);
      if (bases.size() != 6)
        throw std::invalid_argument("CBAM program has malformed base vector");
      bases[4] = fm_base;
      result["base_addresses"] = bases;
      return result;
    };

    std::lock_guard<std::mutex> guard(schedule_mutex_);
    const auto composite_begin = std::chrono::steady_clock::now();
    py::list timings;
    auto read_reduce_vector = [&](uint64_t base, uint64_t output_addr) {
      py::list requests;
      const uint64_t address = (base + output_addr) * 128;
      for (int bank = 0; bank < 2; ++bank)
        requests.append(py::make_tuple(
            bank, address + (bank ? 0x100000000ULL : 0ULL), reduce_output_size));
      return c2h_batch_impl(requests, true);
    };
    auto exact_span_join = [&](const py::list &vectors) {
      if (vectors.size() != 4)
        throw std::runtime_error("CBAM reducer returned malformed vector set");
      py::tuple joined(2);
      for (size_t bank = 0; bank < 2; ++bank) {
        auto left = py::array_t<uint8_t, py::array::c_style |
                               py::array::forcecast>::ensure(vectors[bank]);
        auto right = py::array_t<uint8_t, py::array::c_style |
                                py::array::forcecast>::ensure(vectors[2 + bank]);
        if (!left || !right || static_cast<size_t>(left.size()) != reduce_output_size ||
            static_cast<size_t>(right.size()) != reduce_output_size)
          throw std::runtime_error("CBAM reducer vector size mismatch");
        py::array_t<uint8_t> output(fc_input_size);
        std::memcpy(output.mutable_data(), left.data(), reduce_output_size);
        std::memcpy(output.mutable_data() + reduce_output_size,
                    right.data(), reduce_output_size);
        joined[bank] = std::move(output);
      }
      ++cbam_exact_span_joins_;
      cbam_exact_span_bytes_ += fc_input_size * 2;
      return joined;
    };
    auto upload_fc_input = [&](const py::tuple &pair, uint64_t base) {
      py::list requests;
      const uint64_t address = (base + fc_input) * 128;
      for (int bank = 0; bank < 2; ++bank)
        requests.append(py::make_tuple(
            bank, address + (bank ? 0x100000000ULL : 0ULL), pair[bank]));
      write_reg(0x34, 1);
      h2c_batch_impl(requests, true);
    };

    py::list avg_vectors, max_vectors;
    for (size_t segment = 0; segment < 2; ++segment) {
      py::list avg_requests;
      const uint64_t avg_address = (avg_reduce_base + reduce_input) * 128;
      for (int bank = 0; bank < 2; ++bank)
        avg_requests.append(py::make_tuple(
            bank, avg_address + (bank ? 0x100000000ULL : 0ULL),
            sources[segment][bank]));
      write_reg(0x34, 1);
      h2c_batch_impl(avg_requests, true);
      timings.append(launch_program(
          with_fm_base(avg_program, avg_reduce_base), timeout_ms));
      py::list average = read_reduce_vector(avg_reduce_base, reduce_output);
      avg_vectors.append(average[0]); avg_vectors.append(average[1]);

      py::list max_requests;
      const uint64_t max_address = (max_reduce_base + reduce_input) * 128;
      for (int bank = 0; bank < 2; ++bank)
        max_requests.append(py::make_tuple(
            bank, max_address + (bank ? 0x100000000ULL : 0ULL),
            sources[segment][bank]));
      write_reg(0x34, 1);
      h2c_batch_impl(max_requests, true);
      timings.append(launch_program(
          with_fm_base(max_program, max_reduce_base), timeout_ms));
      py::list maximum = read_reduce_vector(max_reduce_base, reduce_output);
      max_vectors.append(maximum[0]); max_vectors.append(maximum[1]);
    }
    py::tuple avg_joined = exact_span_join(avg_vectors);
    py::tuple max_joined = exact_span_join(max_vectors);

    upload_fc_input(avg_joined, avg_fc_base);
    timings.append(launch_program(with_fm_base(fc_program, avg_fc_base), timeout_ms));
    py::list avg_output_requests;
    const uint64_t avg_output_address = (avg_fc_base + fc_output) * 128;
    for (int bank = 0; bank < 2; ++bank)
      avg_output_requests.append(py::make_tuple(
          bank, avg_output_address + (bank ? 0x100000000ULL : 0ULL), fc_output_size));
    py::list avg_outputs = c2h_batch_impl(avg_output_requests, true);

    upload_fc_input(max_joined, max_fc_base);
    timings.append(launch_program(with_fm_base(fc_program, max_fc_base), timeout_ms));
    py::list max_output_requests;
    const uint64_t max_output_address = (max_fc_base + fc_output) * 128;
    for (int bank = 0; bank < 2; ++bank)
      max_output_requests.append(py::make_tuple(
          bank, max_output_address + (bank ? 0x100000000ULL : 0ULL), fc_output_size));
    py::list max_outputs = c2h_batch_impl(max_output_requests, true);
    py::tuple average_pair(2), maximum_pair(2);
    average_pair[0] = avg_outputs[0]; average_pair[1] = avg_outputs[1];
    maximum_pair[0] = max_outputs[0]; maximum_pair[1] = max_outputs[1];
    ++npu_chain_calls_;
    ++cbam_composite_calls_;
    cbam_composite_dispatches_ += 6;
    cbam_composite_seconds_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - composite_begin).count();
    py::dict result;
    result["average_pair"] = std::move(average_pair);
    result["maximum_pair"] = std::move(maximum_pair);
    result["average_joined_pair"] = avg_joined;
    result["maximum_joined_pair"] = max_joined;
    result["timings"] = std::move(timings);
    result["source_workspace_blocks"] = 0;
    result["fc_workspace_blocks"] = 0;
    result["source_h2c_reuse"] = 0;
    result["exact_span_join_location"] = "cpp_pinned_buffer";
    result["workspace_policy"] = "per_operator_compiler_qualified_resident_fm";
    result["generic_workspace_base_blocks"] = workspace;
    return result;
  }

  py::object pack_int8_nchw(const py::array_t<int8_t,
                            py::array::c_style | py::array::forcecast> &input,
                            size_t half_size) {
    py::list inputs;
    inputs.append(input);
    return pack_int8_nchw_segments(inputs, half_size);
  }

  py::object pack_int8_nchw_segments(const py::list &input_list,
                                     size_t half_size) {
    if (input_list.empty()) throw std::invalid_argument("INT8 segment list is empty");
    std::vector<py::array_t<int8_t, py::array::c_style | py::array::forcecast>> inputs;
    struct InputView { const int8_t *data; NchwShape shape; size_t channel_offset; };
    std::vector<InputView> views;
    NchwShape shape{};
    size_t channels = 0;
    for (const py::handle &handle : input_list) {
      auto array = py::array_t<int8_t,
          py::array::c_style | py::array::forcecast>::ensure(handle);
      if (!array) throw std::invalid_argument("INT8 segment must be contiguous NCHW");
      const NchwShape current = nchw_shape(array.request());
      if (inputs.empty()) shape = current;
      if (current.n != shape.n || current.h != shape.h || current.w != shape.w)
        throw std::invalid_argument("INT8 segments must share N/H/W");
      channels += current.c;
      inputs.push_back(std::move(array));
      views.push_back(InputView{inputs.back().data(), current, channels - current.c});
    }
    shape.c = channels;
    const std::string kind = layout_kind(shape, half_size * 2);
    if (kind.empty()) return py::none();
    py::array_t<uint8_t> even(half_size), odd(half_size);
    std::memset(even.mutable_data(), 0, half_size);
    std::memset(odd.mutable_data(), 0, half_size);
    auto *even_data = even.mutable_data();
    auto *odd_data = odd.mutable_data();
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto process = [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
          const size_t n = row / shape.h;
          const size_t h = row % shape.h;
          for (const InputView &view : views) {
            const NchwShape &current = view.shape;
            const int8_t *source = view.data;
          for (size_t c = 0; c < current.c; ++c) {
              const size_t global_c = view.channel_offset + c;
              for (size_t w = 0; w < shape.w; ++w) {
                const size_t source_index = ((n * current.c + c) * shape.h + h) * shape.w + w;
                uint8_t *bank = ((w % 16) < 8) ? even_data : odd_data;
                size_t target_index = 0;
                if (kind == "normal16") {
                  target_index = normal16_index(shape, n, global_c, h, w);
                } else {
                  target_index = compact4_index(shape, n, global_c, h, w);
                }
                bank[target_index] = static_cast<uint8_t>(source[source_index]);
              }
            }
          }
        }
      };
      parallel_rows(shape.n * shape.h, shape.c * shape.w, process);
    }
    const auto end = std::chrono::steady_clock::now();
    ++codec_pack_calls_;
    codec_pack_bytes_ += half_size * 2;
    codec_pack_seconds_ += std::chrono::duration<double>(end - begin).count();
    return py::make_tuple(std::move(even), std::move(odd), kind);
  }

  py::object unpack_int8_nchw(
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &even,
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &odd,
      const std::vector<size_t> &dimensions) {
    if (dimensions.size() != 4)
      throw std::invalid_argument("INT8 unpack requires four dimensions");
    const NchwShape shape{dimensions[0], dimensions[1], dimensions[2], dimensions[3]};
    if (even.size() != odd.size())
      throw std::invalid_argument("DDR output banks have different sizes");
    const std::string kind = layout_kind(
        shape, static_cast<size_t>(even.size() + odd.size()));
    if (kind.empty()) return py::none();
    py::array_t<int8_t> output({shape.n, shape.c, shape.h, shape.w});
    const uint8_t *even_data = even.data();
    const uint8_t *odd_data = odd.data();
    int8_t *target = output.mutable_data();
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto process = [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
          const size_t n = row / shape.h;
          const size_t h = row % shape.h;
          for (size_t c = 0; c < shape.c; ++c) {
            for (size_t w = 0; w < shape.w; ++w) {
              const uint8_t *bank = ((w % 16) < 8) ? even_data : odd_data;
              size_t source_index = 0;
              if (kind == "normal16") {
                source_index = normal16_index(shape, n, c, h, w);
              } else {
                source_index = compact4_index(shape, n, c, h, w);
              }
              const size_t target_index = ((n * shape.c + c) * shape.h + h) * shape.w + w;
              target[target_index] = static_cast<int8_t>(bank[source_index]);
            }
          }
        }
      };
      parallel_rows(shape.n * shape.h, shape.c * shape.w, process);
    }
    const auto end = std::chrono::steady_clock::now();
    ++codec_unpack_calls_;
    codec_unpack_bytes_ += static_cast<size_t>(even.size() + odd.size());
    codec_unpack_seconds_ += std::chrono::duration<double>(end - begin).count();
    return py::make_tuple(std::move(output), kind);
  }

  py::object interleave_polyphase_normal16(
      const py::list &phase_bank_pairs,
      const std::vector<size_t> &phase_dimensions) {
    if (phase_bank_pairs.size() != 4)
      throw std::invalid_argument("polyphase interleave requires four phase pairs");
    if (phase_dimensions.size() != 4)
      throw std::invalid_argument("polyphase interleave requires NCHW dimensions");
    const NchwShape source{phase_dimensions[0], phase_dimensions[1],
                           phase_dimensions[2], phase_dimensions[3]};
    if (source.n == 0 || source.c == 0 || source.h == 0 || source.w == 0)
      throw std::invalid_argument("polyphase interleave does not accept empty tensors");
    const size_t source_half_size = source.n * source.h * ceil_div(source.w, 16) *
                                    ceil_div(source.c, 16) * 128;
    std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>> banks;
    banks.reserve(8);
    for (const py::handle &handle : phase_bank_pairs) {
      const py::tuple pair = py::cast<py::tuple>(handle);
      if (pair.size() != 2)
        throw std::invalid_argument("each polyphase item must be (even,odd)");
      for (const py::handle &bank_handle : pair) {
        auto bank = py::array_t<uint8_t, py::array::c_style |
                                py::array::forcecast>::ensure(bank_handle);
        if (!bank || static_cast<size_t>(bank.size()) != source_half_size)
          throw std::invalid_argument("polyphase source is not a normal16 bank");
        banks.push_back(std::move(bank));
      }
    }

    const NchwShape target{source.n, source.c, source.h * 2, source.w * 2};
    const size_t target_half_size = target.n * target.h * ceil_div(target.w, 16) *
                                    ceil_div(target.c, 16) * 128;
    py::array_t<uint8_t> even(target_half_size), odd(target_half_size);
    std::memset(even.mutable_data(), 0, target_half_size);
    std::memset(odd.mutable_data(), 0, target_half_size);
    uint8_t *target_banks[2] = {even.mutable_data(), odd.mutable_data()};
    std::array<const uint8_t *, 8> source_banks{};
    for (size_t index = 0; index < banks.size(); ++index)
      source_banks[index] = banks[index].data();
    const size_t source_w_blocks = ceil_div(source.w, 16);
    const size_t target_w_blocks = ceil_div(target.w, 16);
    const size_t channel_blocks = ceil_div(source.c, 16);
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto process = [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
          const size_t n = row / source.h;
          const size_t y = row % source.h;
          for (size_t c = 0; c < source.c; ++c) {
            for (size_t x = 0; x < source.w; ++x) {
              const size_t source_bank = (x % 16) < 8 ? 0 : 1;
              const size_t source_index =
                  (((((n * source.h + y) * source_w_blocks + x / 16) *
                       channel_blocks + c / 16) * 8 + x % 8) * 16 + c % 16);
              for (size_t phase_y = 0; phase_y < 2; ++phase_y) {
                for (size_t phase_x = 0; phase_x < 2; ++phase_x) {
                  const size_t phase = phase_y * 2 + phase_x;
                  const size_t output_y = y * 2 + phase_y;
                  const size_t output_x = x * 2 + phase_x;
                  const size_t output_bank = (output_x % 16) < 8 ? 0 : 1;
                  const size_t output_index =
                      (((((n * target.h + output_y) * target_w_blocks +
                           output_x / 16) * channel_blocks + c / 16) * 8 +
                         output_x % 8) * 16 + c % 16);
                  target_banks[output_bank][output_index] =
                      source_banks[phase * 2 + source_bank][source_index];
                }
              }
            }
          }
        }
      };
      parallel_rows(source.n * source.h, source.c * source.w * 4, process);
    }
    const auto end = std::chrono::steady_clock::now();
    ++polyphase_interleave_calls_;
    polyphase_interleave_bytes_ += source_half_size * 8 + target_half_size * 2;
    polyphase_interleave_seconds_ +=
        std::chrono::duration<double>(end - begin).count();
    py::list shape;
    shape.append(target.n); shape.append(target.c);
    shape.append(target.h); shape.append(target.w);
    return py::make_tuple(std::move(even), std::move(odd), std::move(shape),
                          "normal16");
  }

  py::object requantize_normal16(
      const py::tuple &source_pair, const std::vector<size_t> &dimensions,
      double source_multiplier, double target_multiplier) {
    if (dimensions.size() != 4)
      throw std::invalid_argument("normal16 requantize requires NCHW dimensions");
    const NchwShape shape{dimensions[0], dimensions[1], dimensions[2], dimensions[3]};
    if (shape.n == 0 || shape.c == 0 || shape.h == 0 || shape.w == 0)
      throw std::invalid_argument("normal16 requantize does not accept empty tensors");
    if (!(source_multiplier > 0.0) || !(target_multiplier > 0.0))
      throw std::invalid_argument("normal16 requantize multipliers must be positive");
    auto source = normal16_pair(source_pair, shape, "requantize source");
    const size_t half_size = normal16_half_size(shape);
    py::array_t<uint8_t> even(half_size), odd(half_size);
    const double ratio = target_multiplier / source_multiplier;
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto process = [&](size_t bank_begin, size_t bank_end) {
        for (size_t bank = bank_begin; bank < bank_end; ++bank) {
          const int8_t *input = reinterpret_cast<const int8_t *>(source[bank].data());
          int8_t *output = reinterpret_cast<int8_t *>(
              bank == 0 ? even.mutable_data() : odd.mutable_data());
          for (size_t index = 0; index < half_size; ++index) {
            // NumPy/PyTorch quantization uses round-to-nearest-even.  Keep the
            // raw-layout carrier bit-exact at half-way values as well.
            const long quantized = static_cast<long>(
                std::nearbyint(static_cast<double>(input[index]) * ratio));
            output[index] = static_cast<int8_t>(std::max<long>(
                -128, std::min<long>(127, quantized)));
          }
        }
      };
      parallel_rows(2, half_size, process);
    }
    const auto end = std::chrono::steady_clock::now();
    ++normal16_requantize_calls_;
    normal16_requantize_bytes_ += half_size * 4;
    normal16_requantize_seconds_ +=
        std::chrono::duration<double>(end - begin).count();
    return py::make_tuple(std::move(even), std::move(odd), "normal16");
  }

  py::list crop_normal16_tiles(
      const py::tuple &source_pair, const std::vector<size_t> &source_dimensions,
      const std::vector<size_t> &grid_hw,
      const std::vector<size_t> &core_hw,
      const std::vector<size_t> &physical_hw,
      const std::vector<size_t> &halo_tblr) {
    if (source_dimensions.size() != 4 || grid_hw.size() != 2 ||
        core_hw.size() != 2 || physical_hw.size() != 2 || halo_tblr.size() != 4)
      throw std::invalid_argument("malformed normal16 crop contract");
    const NchwShape source_shape{source_dimensions[0], source_dimensions[1],
                                 source_dimensions[2], source_dimensions[3]};
    const size_t grid_h = grid_hw[0], grid_w = grid_hw[1];
    const size_t core_h = core_hw[0], core_w = core_hw[1];
    const size_t top = halo_tblr[0], bottom = halo_tblr[1];
    const size_t left = halo_tblr[2], right = halo_tblr[3];
    const NchwShape tile_shape{source_shape.n, source_shape.c,
                               physical_hw[0], physical_hw[1]};
    if (grid_h == 0 || grid_w == 0 || core_h == 0 || core_w == 0 ||
        tile_shape.h != top + core_h + bottom ||
        tile_shape.w != left + core_w + right ||
        grid_h * core_h < source_shape.h || grid_w * core_w < source_shape.w)
      throw std::invalid_argument("inconsistent normal16 crop geometry");
    auto source = normal16_pair(source_pair, source_shape, "crop source");
    const size_t tile_half_size = normal16_half_size(tile_shape);
    py::list result;
    const auto begin = std::chrono::steady_clock::now();
    for (size_t tile_y = 0; tile_y < grid_h; ++tile_y) {
      for (size_t tile_x = 0; tile_x < grid_w; ++tile_x) {
        py::array_t<uint8_t> even(tile_half_size), odd(tile_half_size);
        std::memset(even.mutable_data(), 0, tile_half_size);
        std::memset(odd.mutable_data(), 0, tile_half_size);
        uint8_t *targets[2] = {even.mutable_data(), odd.mutable_data()};
        {
          py::gil_scoped_release release;
          auto process = [&](size_t row_begin, size_t row_end) {
            for (size_t row = row_begin; row < row_end; ++row) {
              const size_t n = row / tile_shape.h;
              const size_t y = row % tile_shape.h;
              const int64_t source_y = static_cast<int64_t>(tile_y * core_h + y) -
                                       static_cast<int64_t>(top);
              if (source_y < 0 || source_y >= static_cast<int64_t>(source_shape.h))
                continue;
              for (size_t c = 0; c < source_shape.c; ++c) {
                for (size_t x = 0; x < tile_shape.w; ++x) {
                  const int64_t source_x =
                      static_cast<int64_t>(tile_x * core_w + x) -
                      static_cast<int64_t>(left);
                  if (source_x < 0 ||
                      source_x >= static_cast<int64_t>(source_shape.w)) continue;
                  const size_t sx = static_cast<size_t>(source_x);
                  const size_t sy = static_cast<size_t>(source_y);
                  const size_t source_bank = (sx % 16) < 8 ? 0 : 1;
                  const size_t target_bank = (x % 16) < 8 ? 0 : 1;
                  targets[target_bank][normal16_index(tile_shape, n, c, y, x)] =
                      source[source_bank].data()[normal16_index(
                          source_shape, n, c, sy, sx)];
                }
              }
            }
          };
          parallel_rows(tile_shape.n * tile_shape.h,
                        tile_shape.c * tile_shape.w, process);
        }
        result.append(py::make_tuple(std::move(even), std::move(odd)));
      }
    }
    const auto end = std::chrono::steady_clock::now();
    ++normal16_crop_calls_;
    normal16_crop_tiles_ += grid_h * grid_w;
    normal16_crop_seconds_ += std::chrono::duration<double>(end - begin).count();
    return result;
  }

  py::object scatter_normal16_tiles(
      const py::list &tile_pairs, const std::vector<size_t> &tile_dimensions,
      const std::vector<size_t> &target_dimensions,
      const std::vector<size_t> &grid_hw,
      const std::vector<size_t> &core_hw,
      const std::vector<size_t> &crop_yx) {
    if (tile_dimensions.size() != 4 || target_dimensions.size() != 4 ||
        grid_hw.size() != 2 || core_hw.size() != 2 || crop_yx.size() != 2)
      throw std::invalid_argument("malformed normal16 scatter contract");
    const NchwShape tile_shape{tile_dimensions[0], tile_dimensions[1],
                               tile_dimensions[2], tile_dimensions[3]};
    const NchwShape target_shape{target_dimensions[0], target_dimensions[1],
                                 target_dimensions[2], target_dimensions[3]};
    const size_t grid_h = grid_hw[0], grid_w = grid_hw[1];
    const size_t core_h = core_hw[0], core_w = core_hw[1];
    if (tile_pairs.size() != grid_h * grid_w || tile_shape.n != target_shape.n ||
        tile_shape.c != target_shape.c || crop_yx[0] + core_h > tile_shape.h ||
        crop_yx[1] + core_w > tile_shape.w)
      throw std::invalid_argument("inconsistent normal16 scatter geometry");
    std::vector<Normal16Pair> tiles;
    tiles.reserve(tile_pairs.size());
    for (const py::handle &handle : tile_pairs)
      tiles.push_back(normal16_pair(handle, tile_shape, "scatter tile"));
    const size_t target_half_size = normal16_half_size(target_shape);
    py::array_t<uint8_t> even(target_half_size), odd(target_half_size);
    std::memset(even.mutable_data(), 0, target_half_size);
    std::memset(odd.mutable_data(), 0, target_half_size);
    uint8_t *targets[2] = {even.mutable_data(), odd.mutable_data()};
    const auto begin = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto process = [&](size_t target_y_begin, size_t target_y_end) {
        for (size_t target_y = target_y_begin; target_y < target_y_end; ++target_y) {
          const size_t tile_y = target_y / core_h;
          const size_t local_y = crop_yx[0] + target_y % core_h;
          if (tile_y >= grid_h) continue;
          for (size_t n = 0; n < target_shape.n; ++n) {
            for (size_t c = 0; c < target_shape.c; ++c) {
              for (size_t target_x = 0; target_x < target_shape.w; ++target_x) {
                const size_t tile_x = target_x / core_w;
                if (tile_x >= grid_w) continue;
                const size_t local_x = crop_yx[1] + target_x % core_w;
                const size_t source_bank = (local_x % 16) < 8 ? 0 : 1;
                const size_t target_bank = (target_x % 16) < 8 ? 0 : 1;
                const auto &tile = tiles[tile_y * grid_w + tile_x];
                targets[target_bank][normal16_index(
                    target_shape, n, c, target_y, target_x)] =
                    tile[source_bank].data()[normal16_index(
                        tile_shape, n, c, local_y, local_x)];
              }
            }
          }
        }
      };
      parallel_rows(target_shape.h, target_shape.n * target_shape.c * target_shape.w,
                    process);
    }
    const auto end = std::chrono::steady_clock::now();
    ++normal16_scatter_calls_;
    normal16_scatter_tiles_ += grid_h * grid_w;
    normal16_scatter_seconds_ += std::chrono::duration<double>(end - begin).count();
    py::list shape;
    shape.append(target_shape.n); shape.append(target_shape.c);
    shape.append(target_shape.h); shape.append(target_shape.w);
    return py::make_tuple(std::move(even), std::move(odd), std::move(shape),
                          "normal16");
  }

  py::list c2h_batch_impl(const py::list &requests, bool reopen_each_segment) {
    std::array<std::vector<Segment>, 2> segments;
    std::array<size_t, 2> cursor{};
    size_t index = 0;
    for (const py::handle &handle : requests) {
      py::tuple item = py::cast<py::tuple>(handle);
      if (item.size() != 3) throw std::invalid_argument("C2H request must be (bank,address,size)");
      const int bank = py::cast<int>(item[0]);
      if (bank < 0 || bank > 1) throw std::invalid_argument("bank must be 0 or 1");
      const uint64_t address = py::cast<uint64_t>(item[1]);
      const size_t size = py::cast<size_t>(item[2]);
      if (size == 0) throw std::invalid_argument("zero-byte C2H request");
      const size_t offset = cursor[bank];
      cursor[bank] += size;
      output_[bank].reserve(cursor[bank]);
      segments[bank].push_back(Segment{bank, address, size, offset, index++});
    }
    // Both bank halves form one physical tensor transfer.  The current
    // bitstream requires the qualified parallel-bank C2H ordering; serial
    // halves can eventually leave a later program without completion.
    run_parallel([&](int bank) {
      read_segments(bank, segments[bank], reopen_each_segment);
    });
    if (reopen_each_segment) ++safe_c2h_batches_;
    std::vector<py::array_t<uint8_t>> arrays(index);
    for (int bank = 0; bank < 2; ++bank) {
      for (const Segment &segment : segments[bank]) {
        py::array_t<uint8_t> result(segment.size);
        std::memcpy(result.mutable_data(), output_[bank].data() + segment.scratch_offset,
                    segment.size);
        arrays[segment.result_index] = std::move(result);
      }
    }
    py::list result;
    for (auto &array : arrays) result.append(std::move(array));
    ++c2h_batches_;
    c2h_segments_ += requests.size();
    c2h_bytes_ += cursor[0] + cursor[1];
    return result;
  }

  py::dict stats() const {
    py::dict result;
    result["h2c_batches"] = h2c_batches_;
    result["h2c_segments"] = h2c_segments_;
    result["h2c_syscalls"] = h2c_syscalls_.load();
    result["h2c_bytes"] = h2c_bytes_;
    result["c2h_batches"] = c2h_batches_;
    result["c2h_segments"] = c2h_segments_;
    result["c2h_syscalls"] = c2h_syscalls_.load();
    result["c2h_bytes"] = c2h_bytes_;
    result["c2h_bank_policy"] = "always_parallel_hardware_contract";
    result["pinned_h2c_capacity"] = input_[0].capacity() + input_[1].capacity();
    result["pinned_c2h_capacity"] = output_[0].capacity() + output_[1].capacity();
    result["persistent_fds"] = 4;
    result["parallel_banks"] = true;
    result["safe_h2c_batches"] = safe_h2c_batches_;
    result["safe_c2h_batches"] = safe_c2h_batches_;
    result["safe_mode_barrier"] = "open_lseek_transfer_close_per_coalesced_segment";
    result["coalesce_policy"] = "exactly_contiguous_only";
    result["npu_chain_calls"] = npu_chain_calls_;
    result["npu_chain_dispatches"] = npu_chain_dispatches_;
    result["npu_chain_seconds"] = npu_chain_seconds_;
    result["codec_pack_calls"] = codec_pack_calls_;
    result["codec_pack_bytes"] = codec_pack_bytes_;
    result["codec_pack_seconds"] = codec_pack_seconds_;
    result["codec_unpack_calls"] = codec_unpack_calls_;
    result["codec_unpack_bytes"] = codec_unpack_bytes_;
    result["codec_unpack_seconds"] = codec_unpack_seconds_;
    result["polyphase_interleave_calls"] = polyphase_interleave_calls_;
    result["polyphase_interleave_bytes"] = polyphase_interleave_bytes_;
    result["polyphase_interleave_seconds"] = polyphase_interleave_seconds_;
    result["normal16_requantize_calls"] = normal16_requantize_calls_;
    result["normal16_requantize_bytes"] = normal16_requantize_bytes_;
    result["normal16_requantize_seconds"] = normal16_requantize_seconds_;
    result["normal16_crop_calls"] = normal16_crop_calls_;
    result["normal16_crop_tiles"] = normal16_crop_tiles_;
    result["normal16_crop_seconds"] = normal16_crop_seconds_;
    result["normal16_scatter_calls"] = normal16_scatter_calls_;
    result["normal16_scatter_tiles"] = normal16_scatter_tiles_;
    result["normal16_scatter_seconds"] = normal16_scatter_seconds_;
    result["cbam_composite_calls"] = cbam_composite_calls_;
    result["cbam_composite_dispatches"] = cbam_composite_dispatches_;
    result["cbam_composite_seconds"] = cbam_composite_seconds_;
    result["cbam_exact_span_joins"] = cbam_exact_span_joins_;
    result["cbam_exact_span_bytes"] = cbam_exact_span_bytes_;
    result["cbam_join_location"] = "cpp_pinned_buffer";
    return result;
  }

  void reset_stats() {
    // A resident executor reuses file descriptors and pinned buffers across
    // frames, but report/gate counters are per frame.  Reset only accounting;
    // device state, allocations, and DDR contents intentionally remain live.
    std::lock_guard<std::mutex> guard(schedule_mutex_);
    stale_events_ = 0;
    h2c_batches_ = 0;
    h2c_segments_ = 0;
    h2c_syscalls_.store(0);
    h2c_bytes_ = 0;
    c2h_batches_ = 0;
    c2h_segments_ = 0;
    c2h_syscalls_.store(0);
    c2h_bytes_ = 0;
    safe_h2c_batches_ = 0;
    safe_c2h_batches_ = 0;
    npu_chain_calls_ = 0;
    npu_chain_dispatches_ = 0;
    npu_chain_seconds_ = 0.0;
    codec_pack_calls_ = 0;
    codec_pack_bytes_ = 0;
    codec_pack_seconds_ = 0.0;
    codec_unpack_calls_ = 0;
    codec_unpack_bytes_ = 0;
    codec_unpack_seconds_ = 0.0;
    polyphase_interleave_calls_ = 0;
    polyphase_interleave_bytes_ = 0;
    polyphase_interleave_seconds_ = 0.0;
    normal16_requantize_calls_ = 0;
    normal16_requantize_bytes_ = 0;
    normal16_requantize_seconds_ = 0.0;
    normal16_crop_calls_ = 0;
    normal16_crop_tiles_ = 0;
    normal16_crop_seconds_ = 0.0;
    normal16_scatter_calls_ = 0;
    normal16_scatter_tiles_ = 0;
    normal16_scatter_seconds_ = 0.0;
    cbam_composite_calls_ = 0;
    cbam_composite_dispatches_ = 0;
    cbam_composite_seconds_ = 0.0;
    cbam_exact_span_joins_ = 0;
    cbam_exact_span_bytes_ = 0;
  }

 private:
  double launch_program(const py::dict &program, int timeout_ms) {
    const std::string stage = py::cast<std::string>(program["stage_id"]);
    const std::vector<uint64_t> bases =
        py::cast<std::vector<uint64_t>>(program["base_addresses"]);
    const std::vector<uint64_t> ranges =
        py::cast<std::vector<uint64_t>>(program["isa_ranges"]);
    if (bases.size() != 6 || ranges.size() != 2)
      throw std::invalid_argument(stage + ": malformed NPU cfg vectors");
    write_reg(0x2c, 1);
    write_reg(0x00, 0x3f);
    configure(bases, ranges);
    write_reg(0x34, 1);
    clear_interrupt();
    drain_events();
    write_reg(0x34, 2);
    const double elapsed = wait_for_completion(stage, timeout_ms);
    ++npu_chain_dispatches_;
    npu_chain_seconds_ += elapsed;
    return elapsed;
  }

  void write_reg(uint64_t address, uint32_t value) {
    if (address + sizeof(uint32_t) > 4096)
      throw std::runtime_error("BAR register address out of range");
    auto *target = reinterpret_cast<volatile uint32_t *>(
        static_cast<uint8_t *>(user_mapping_) + address);
    *target = value;
  }

  uint32_t read_reg(uint64_t address) const {
    if (address + sizeof(uint32_t) > 4096)
      throw std::runtime_error("BAR register address out of range");
    auto *target = reinterpret_cast<volatile uint32_t *>(
        static_cast<uint8_t *>(user_mapping_) + address);
    return *target;
  }

  void clear_interrupt() {
    if (read_reg(0x00) & 4U) write_reg(0x00, 0x3f);
  }

  void configure(const std::vector<uint64_t> &bases,
                 const std::vector<uint64_t> &ranges) {
    write_reg(0x40, static_cast<uint32_t>(bases[5]));
    write_reg(0x44, static_cast<uint32_t>(bases[0]));
    write_reg(0x48, static_cast<uint32_t>(ranges[1] | (ranges[0] << 16)));
    write_reg(0x90, static_cast<uint32_t>(bases[1]));
    write_reg(0x94, static_cast<uint32_t>(bases[2]));
    write_reg(0x98, static_cast<uint32_t>(bases[3]));
    write_reg(0x9c, static_cast<uint32_t>(bases[4]));
    clear_interrupt();
  }

  void drain_events() {
    while (true) {
      std::array<pollfd, 16> descriptors{};
      for (int index = 0; index < 16; ++index)
        descriptors[index] = pollfd{events_[index], POLLIN | POLLERR | POLLHUP, 0};
      const int result = ::poll(descriptors.data(), descriptors.size(), 0);
      if (result < 0) fail("event drain poll");
      if (result == 0) return;
      bool consumed = false;
      for (int index = 0; index < 16; ++index) {
        if (descriptors[index].revents & (POLLERR | POLLHUP))
          throw std::runtime_error("event fd error while draining");
        if (!(descriptors[index].revents & POLLIN)) continue;
        uint32_t value = 0;
        const ssize_t bytes = ::pread(events_[index], &value, sizeof(value), 0);
        if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (bytes < 0) fail("event drain pread");
        if (bytes != static_cast<ssize_t>(sizeof(value)))
          throw std::runtime_error("short event drain pread");
        consumed = true;
        ++stale_events_;
      }
      if (!consumed) return;
    }
  }

  double wait_for_completion(const std::string &stage, int timeout_ms) {
    const auto begin = std::chrono::steady_clock::now();
    const auto deadline = begin + std::chrono::milliseconds(timeout_ms);
    while (true) {
      std::array<pollfd, 16> descriptors{};
      for (int index = 0; index < 16; ++index)
        descriptors[index] = pollfd{events_[index], POLLIN | POLLERR | POLLHUP, 0};
      const auto now = std::chrono::steady_clock::now();
      int remaining = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - now).count());
      if (remaining < 0) remaining = 0;
      const int result = ::poll(descriptors.data(), descriptors.size(), remaining);
      if (result <= 0) {
        std::ostringstream message;
        message << stage << (result == 0 ? ": NPU timeout" : ": event poll failed")
                << " status=0x" << std::hex << read_reg(0x00);
        throw std::runtime_error(message.str());
      }
      int event_id = -1;
      for (int index = 0; index < 16; ++index) {
        if (descriptors[index].revents & POLLIN) { event_id = index; break; }
        if (descriptors[index].revents & (POLLERR | POLLHUP))
          throw std::runtime_error(stage + ": event fd error");
      }
      if (event_id < 0) continue;
      uint32_t event_value = 0;
      const ssize_t bytes = ::pread(
          events_[event_id], &event_value, sizeof(event_value), 0);
      if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;
      if (bytes < 0) fail("event pread");
      if (bytes != static_cast<ssize_t>(sizeof(event_value)))
        throw std::runtime_error(stage + ": short event pread");
      if (event_id != 0) throw std::runtime_error(stage + ": unexpected event id");
      if (read_reg(0x00) & 4U) break;
      ++stale_events_;
    }
    const auto end = std::chrono::steady_clock::now();
    clear_interrupt();
    if (read_reg(0x00) != 0)
      throw std::runtime_error(stage + ": non-zero NPU status");
    return std::chrono::duration<double>(end - begin).count();
  }

  template <typename Function>
  void run_parallel(Function function) {
    run_banks(function, true);
  }

  template <typename Function>
  void run_banks(Function function, bool parallel) {
    std::exception_ptr left_error;
    {
      py::gil_scoped_release release;
      if (!parallel) {
        function(0);
        function(1);
        return;
      }
      std::thread left([&] {
        try { function(0); } catch (...) { left_error = std::current_exception(); }
      });
      try {
        function(1);
      } catch (...) {
        left.join();
        throw;
      }
      left.join();
    }
    if (left_error) std::rethrow_exception(left_error);
  }

  void write_segments(int bank, const std::vector<Segment> &segments,
                      bool reopen_each_segment) {
    std::lock_guard<std::mutex> guard(h2c_mutex_[bank]);
    for (size_t index = 0; index < segments.size();) {
      const Segment &first = segments[index];
      size_t bytes = first.size;
      size_t next = index + 1;
      while (next < segments.size() &&
             segments[next].address == first.address + bytes &&
             segments[next].scratch_offset == first.scratch_offset + bytes) {
        bytes += segments[next].size;
        ++next;
      }
      int fd = h2c_[bank];
      if (reopen_each_segment) {
        fd = ::open(kH2C[bank], O_RDWR);
        if (fd < 0) fail(std::string("open ") + kH2C[bank]);
      }
      try {
        seek_write_exact(fd, input_[bank].data() + first.scratch_offset,
                         bytes, first.address);
      } catch (...) {
        if (reopen_each_segment) ::close(fd);
        throw;
      }
      if (reopen_each_segment) ::close(fd);
      ++h2c_syscalls_;
      index = next;
    }
  }

  void read_segments(int bank, const std::vector<Segment> &segments,
                     bool reopen_each_segment) {
    std::lock_guard<std::mutex> guard(c2h_mutex_[bank]);
    for (size_t index = 0; index < segments.size();) {
      const Segment &first = segments[index];
      size_t bytes = first.size;
      size_t next = index + 1;
      while (next < segments.size() &&
             segments[next].address == first.address + bytes &&
             segments[next].scratch_offset == first.scratch_offset + bytes) {
        bytes += segments[next].size;
        ++next;
      }
      int fd = c2h_[bank];
      if (reopen_each_segment) {
        fd = ::open(kC2H[bank], O_RDWR | O_NONBLOCK);
        if (fd < 0) fail(std::string("open ") + kC2H[bank]);
      }
      try {
        seek_read_exact(fd, output_[bank].data() + first.scratch_offset,
                        bytes, first.address);
      } catch (...) {
        if (reopen_each_segment) ::close(fd);
        throw;
      }
      if (reopen_each_segment) ::close(fd);
      ++c2h_syscalls_;
      index = next;
    }
  }

  int h2c_[2]{-1, -1};
  int c2h_[2]{-1, -1};
  LockedBuffer input_[2];
  LockedBuffer output_[2];
  std::mutex h2c_mutex_[2];
  std::mutex c2h_mutex_[2];
  std::mutex schedule_mutex_;
  int user_fd_ = -1;
  void *user_mapping_ = MAP_FAILED;
  int events_[16]{-1, -1, -1, -1, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, -1};
  uint64_t stale_events_ = 0;
  uint64_t h2c_batches_ = 0;
  uint64_t h2c_segments_ = 0;
  std::atomic<uint64_t> h2c_syscalls_{0};
  uint64_t h2c_bytes_ = 0;
  uint64_t c2h_batches_ = 0;
  uint64_t c2h_segments_ = 0;
  std::atomic<uint64_t> c2h_syscalls_{0};
  uint64_t c2h_bytes_ = 0;
  uint64_t safe_h2c_batches_ = 0;
  uint64_t safe_c2h_batches_ = 0;
  uint64_t npu_chain_calls_ = 0;
  uint64_t npu_chain_dispatches_ = 0;
  double npu_chain_seconds_ = 0.0;
  uint64_t codec_pack_calls_ = 0;
  uint64_t codec_pack_bytes_ = 0;
  double codec_pack_seconds_ = 0.0;
  uint64_t codec_unpack_calls_ = 0;
  uint64_t codec_unpack_bytes_ = 0;
  double codec_unpack_seconds_ = 0.0;
  uint64_t polyphase_interleave_calls_ = 0;
  uint64_t polyphase_interleave_bytes_ = 0;
  double polyphase_interleave_seconds_ = 0.0;
  uint64_t normal16_requantize_calls_ = 0;
  uint64_t normal16_requantize_bytes_ = 0;
  double normal16_requantize_seconds_ = 0.0;
  uint64_t normal16_crop_calls_ = 0;
  uint64_t normal16_crop_tiles_ = 0;
  double normal16_crop_seconds_ = 0.0;
  uint64_t normal16_scatter_calls_ = 0;
  uint64_t normal16_scatter_tiles_ = 0;
  double normal16_scatter_seconds_ = 0.0;
  uint64_t cbam_composite_calls_ = 0;
  uint64_t cbam_composite_dispatches_ = 0;
  double cbam_composite_seconds_ = 0.0;
  uint64_t cbam_exact_span_joins_ = 0;
  uint64_t cbam_exact_span_bytes_ = 0;
};
}  // namespace

PYBIND11_MODULE(fpgaDmaBatch, module) {
  py::class_<DmaBatch>(module, "DmaBatch")
      .def(py::init<>())
      .def_static("validate_descriptor", &normalized_descriptor,
                  py::arg("descriptor"),
                  "Validate and normalize a native tensor layout descriptor")
      .def_static("pack_tensor", &pack_tensor, py::arg("array").noconvert(),
                  py::arg("descriptor"), "Pack a logical tensor into exact DDR bank arrays")
      .def_static("unpack_tensor", &unpack_tensor, py::arg("even").noconvert(),
                  py::arg("odd").noconvert(), py::arg("descriptor"),
                  "Unpack exact DDR bank arrays into a logical tensor")
      .def("h2c_batch", &DmaBatch::h2c_batch,
           "Transfer a batch of (bank,address,uint8-array) segments")
      .def("h2c_batch_safe", &DmaBatch::h2c_batch_safe,
           "Batched parallel-bank H2C with an open/close barrier per segment")
      .def("c2h_batch", &DmaBatch::c2h_batch,
           "Transfer a batch of (bank,address,size) segments")
      .def("c2h_batch_safe", &DmaBatch::c2h_batch_safe,
           "Batched parallel-bank C2H with an open/close barrier per segment")
      .def("run_npu_chain", &DmaBatch::run_npu_chain,
           py::arg("programs"), py::arg("timeout_ms"),
           "Configure and launch a DDR-aliased NPU program chain in C++")
      .def("run_cbam_fused_pool", &DmaBatch::run_cbam_fused_pool,
           py::arg("source_pairs"), py::arg("avg_program"),
           py::arg("max_program"), py::arg("fc_program"),
           py::arg("layout"), py::arg("timeout_ms"),
           "Run segmented fused CBAM avg/max pooling and FC1 in one C++ transaction")
      .def("pack_int8_nchw", &DmaBatch::pack_int8_nchw,
           py::arg("input"), py::arg("half_size"),
           "Pack one NCHW INT8 tensor into the two DS DDR banks")
      .def("pack_int8_nchw_segments", &DmaBatch::pack_int8_nchw_segments,
           py::arg("inputs"), py::arg("half_size"),
           "Pack channel segments directly into one DS DDR tensor")
      .def("unpack_int8_nchw", &DmaBatch::unpack_int8_nchw,
           py::arg("even"), py::arg("odd"), py::arg("shape"),
           "Unpack two DS DDR banks into a contiguous NCHW INT8 tensor")
      .def("interleave_polyphase_normal16", &DmaBatch::interleave_polyphase_normal16,
           py::arg("phase_bank_pairs"), py::arg("phase_shape"),
           "Interleave four normal16 phase tensors directly into normal16 DDR banks")
      .def("requantize_normal16", &DmaBatch::requantize_normal16,
           py::arg("source_pair"), py::arg("dimensions"),
           py::arg("source_multiplier"), py::arg("target_multiplier"),
           "Requantize an INT8 normal16 tensor without unpacking it to NCHW")
      .def("crop_normal16_tiles", &DmaBatch::crop_normal16_tiles,
           py::arg("source_pair"), py::arg("source_dimensions"),
           py::arg("grid_hw"), py::arg("core_hw"), py::arg("physical_hw"),
           py::arg("halo_tblr"),
           "Crop and zero-pad a normal16 tensor into fixed tiles")
      .def("scatter_normal16_tiles", &DmaBatch::scatter_normal16_tiles,
           py::arg("tile_pairs"), py::arg("tile_dimensions"),
           py::arg("target_dimensions"), py::arg("grid_hw"),
           py::arg("core_hw"), py::arg("crop_yx"),
           "Crop tile cores and scatter them into a normal16 tensor")
      .def("stats", &DmaBatch::stats)
      .def("reset_stats", &DmaBatch::reset_stats,
           "Reset per-frame counters while retaining resident device state");
  module.def("_test_fp32_to_bf16", &test_fp32_to_bf16, py::arg("value"));
}
